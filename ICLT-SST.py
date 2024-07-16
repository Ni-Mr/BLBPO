import argparse
import os
import pickle
import ast
import torch
import wandb
from torch.nn import CrossEntropyLoss
import uuid
from torch.optim import Adam
from utils.llm import LlmModel, constrainScoreByWholeExact
from utils.log import Logger
import random
from tqdm import tqdm

logger = Logger(__name__)
logger.set_level(logger.INFO)
logger.write('ICLT-SST.txt')


def get_prompt(sent):
    input_format = 'How is the sentiment of the sentence [sent]? Note: [note]. Respond ONLY with "Great" or "Terrible". Answer: '
    prompt = 'Question: ' + input_format
    if '[sent]' in prompt:
        prompt = prompt.replace('[sent]', f'\"{sent}\"')
    return prompt


def data_loader(filepath, k_shot=0):
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    data_ls = []
    for it in zip(loaded_data['sent'], loaded_data['rel']):
        data_ls.append(it)
    random.shuffle(data_ls)
    data_item = []
    if k_shot:
        shot = k_shot
        for it in data_ls:
            if it[-1] == 'Terrible':
                data_item.append(it)
                shot -= 1
            if shot == 0:
                break
        shot = k_shot
        for it in data_ls:
            if it[-1] == 'Great':
                data_item.append(it)
                shot -= 1
            if shot == 0:
                break
    else:
        data_item = data_ls
    result = {'inputs': [], 'labels': []}
    for it in tqdm(data_item):
        result["inputs"].append(get_prompt(sent=it[0]))
        result['labels'].append(it[-1])
    return result


def example_loader(filepath):
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def vocabulary_loader(filepath):
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def probs_loader(filepath):
    loaded_data = torch.load(filepath)
    return loaded_data


if __name__ == "__main__":
    # 配置参数
    parser = argparse.ArgumentParser(description="Black-Box Tuning for Large-Language-Model")
    parser.add_argument("--experiment_name", type=str, default='ICLT-SST', help="实验名称")
    parser.add_argument("--seed", type=int, default=666, help="A seed for shuffle.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to run wandb.")
    parser.add_argument("--label_to_id", default={"Terrible": 0, "Great": 1}, type=ast.literal_eval)
    parser.add_argument("--train_data_path", type=str, default='data/SST/train.pt', help="The path of the train data.")
    parser.add_argument("--test_data_path", type=str, default='data/SST/test.pt', help="The path of the test data.")
    parser.add_argument("--example_data_path", type=str, default='data/SST/example.pt',
                        help="The path of the example data.")
    parser.add_argument("--llm_service", type=str, default='openai', help="The LLM service, openai, tongyi, zhipuai")
    parser.add_argument("--llm_key", type=str, default='lm-studio',
                        help="The LLM service api-key")
    parser.add_argument("--k_shot_train", default=10, type=int, help="每种类别参与训练的数据量")
    parser.add_argument("--k_shot_test", default=0, type=int, help="每种类别参与测试的数据量，0 denotes full-shot")
    parser.add_argument("--train_epochs", type=int, default=300, help="Total number of training epochs to perform.")
    parser.add_argument("--save_ckpt_path", type=str, default="ckpt/ICLT/SST", help="The path of the checkpoint params.")
    parser.add_argument("--use_ckpt", type=bool, default=False, help="If True, will use checkpoint.")
    parser.add_argument("--use_ckpt_path", type=str, default="", help="The path of the checkpoint params.")
    parser.add_argument("--just_test", type=bool, default=False, help="If True, will just test.")
    parser.add_argument("--test_ckpt_path", type=str, default="ckpt/ICLT/SST/",
                        help="The path of the checkpoint params.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="优化器学习率")
    parser.add_argument("--sample_size", type=int, default=5, help="求解策略梯度时的采样次数")
    parser.add_argument("--example_number", type=int, default=2, help="每种类别被选择的样本的数量")
    parser.add_argument("--example_search_space", type=int, default=100, help="example的搜索空间")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--best_accuracy", default=-float('inf'), type=float)
    parser.add_argument("--experiment_id", default=str(uuid.uuid4()), type=str)
    args = parser.parse_args()
    random.seed(args.seed)
    if args.use_wandb:
        wandb_config = {k: getattr(args, k) for k in vars(args)}
        wandb.init(config=wandb_config, project="blackbox_prompt_tuning", group=args.experiment_name)
    model = LlmModel(model_name=args.llm_service, api_key=args.llm_key)
    model.set_params(args.label_to_id)
    if not args.just_test:
        ce_loss = CrossEntropyLoss()
        train_data = data_loader(args.train_data_path, args.k_shot_train)
        eval_data = train_data
        example_data = example_loader(args.example_data_path)
        if not args.use_ckpt:
            example_probs = torch.FloatTensor(
                [[1 / args.example_search_space] * args.example_search_space] * args.example_number * 2)
            example_probs.requires_grad = True
            optimizer = Adam([{
                "params": [example_probs],
                "weight_decay": args.weight_decay,  # 表示权重衰减，L2正则化项的系数，用于防止过拟合
                "lr": args.learning_rate
            }, ])
        else:
            if not os.path.exists(args.use_ckpt_path):
                logger.error(f"{args.use_ckpt_path} not exists")
            example_probs = torch.FloatTensor(probs_loader(f'{args.use_ckpt_path}/example_probs_best.pth').tolist())
            example_probs.requires_grad = True
            optimizer = Adam([{
                "params": [example_probs],
                "weight_decay": args.weight_decay,  # 表示权重衰减，L2正则化项的系数，用于防止过拟合
                "lr": args.learning_rate
            }, ])
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.train_epochs}")
        logger.info(f"  Train data size = {len(train_data['inputs'])}")
        logger.info(f"  Eval data size = {len(eval_data['inputs'])}")
        logger.info(f"  Example data size = {args.example_search_space}")
        logger.info(f"  Gradient Accumulation steps = {args.sample_size}")
        logger.info(f"  Number of example = {args.example_number * 2}")
        for epoch in range(args.train_epochs):
            example_dist = torch.distributions.Categorical(example_probs)
            examples_discrete_indices_list = []
            loss_list = []
            for k in range(args.sample_size):
                # 进行一次采样
                examples_discrete_indices = example_dist.sample()
                examples_discrete_indices_list.append(examples_discrete_indices)
                indices_list_sample = examples_discrete_indices.int().tolist()
                examples_discrete_list = []
                for idx in range(2):
                    for idy in range(args.example_number):
                        index = indices_list_sample[idx * args.example_number + idy]
                        examples_discrete_list.append(example_data[list(args.label_to_id.keys())[idx]][index])
                # 本次采样组成的提示语
                examples_discrete = '\n'.join(examples_discrete_list) + '\n'
                # 追加了采样后的内容后形成的新question
                step_question = []
                for i in range(len(train_data['inputs'])):
                    step_question.append(
                        examples_discrete.replace('[note]', '') + train_data['inputs'][
                            i].replace('[note]', ''))

                label_keys = list(args.label_to_id.keys())
                converted_target = torch.tensor([args.label_to_id[label] for label in train_data['labels']])
                pred, logits = model.get_pred_result(step_question)
                loss = ce_loss(logits.view(-1, len(label_keys)), converted_target)
                loss_list.append(loss.item())
            loss_avg = sum(loss_list) / args.sample_size  # 用一个batch中的所有样本计算loss
            logger.info(f"train_loss: {loss_avg}, epoch: {epoch}")
            if args.use_wandb:
                wandb.log({'train_loss': loss_avg, 'epoch': epoch})
            # 清零梯度，为优化变量手动计算梯度
            optimizer.zero_grad()
            derivative = (-1 / (example_probs + 1e-6)).repeat(args.sample_size, 1, 1)
            # derivative梯度矩阵，为对角线上取倒数，非对角线上-1*倒数
            for k, examples_discrete_indices in enumerate(examples_discrete_indices_list):
                for i in range(2 * args.example_number):
                    derivative[k][i][examples_discrete_indices[i]] *= -1
            # 梯度估计
            example_probs.grad = torch.zeros_like(example_probs)
            for k in range(args.sample_size):
                example_probs.grad += 1 / (args.sample_size - 1 + 1e-6) * (loss_list[k] - loss_avg) * derivative[k]
            # 完成一次参数更新
            optimizer.step()
            # detach可以分离梯度对数值进行计算
            constrainScoreByWholeExact(example_probs.detach())
            eval_example = []
            for idx in range(2):
                for idy in range(args.example_number):
                    index = example_probs.argmax(1)[idx * args.example_number + idy]
                    eval_example.append(example_data[list(args.label_to_id.keys())[idx]][index])
            eval_example = examples_discrete = '\n'.join(eval_example) + '\n'
            accuracy = model.eval(eval_data, example=eval_example)
            if accuracy >= args.best_accuracy:
                args.best_accuracy = accuracy
                probs_path = os.path.join(args.save_ckpt_path, args.experiment_id)
                if not os.path.exists(probs_path):
                    os.makedirs(probs_path)
                torch.save(example_probs, f"{probs_path}/example_probs_best.pth")
                if args.use_wandb:
                    wandb.alert(title='best_eval', text=f'epoch: {epoch}, accuracy: {accuracy}')
            else:
                probs_path = os.path.join(args.save_ckpt_path, args.experiment_id)
                if not os.path.exists(probs_path):
                    os.makedirs(probs_path)
                torch.save(example_probs, f"{probs_path}/example_probs_latest.pth")
            logger.info(f"eval-accuracy: {accuracy}, epoch: {epoch}")
            if args.use_wandb:
                wandb.log({'eval-accuracy': accuracy, 'epoch': epoch})
    else:
        if not os.path.exists(args.test_ckpt_path):
            logger.error(f"{args.test_ckpt_path} not exists")
        example_probs = torch.FloatTensor(probs_loader(f'{args.test_ckpt_path}/example_probs_best.pth').tolist())
        test_data = data_loader(args.test_data_path, args.k_shot_test)
        example_data = example_loader(args.example_data_path)
        logger.info("***** Running testing *****")
        logger.info(f"  Test data size = {len(test_data['inputs'])}")
        # 对本次更新结果进行评估
        test_example = []
        for idx in range(2):
            for idy in range(args.example_number):
                index = example_probs.argmax(1)[idx * args.example_number + idy]
                test_example.append(example_data[list(args.label_to_id.keys())[idx]][index])
        test_example = examples_discrete = '\n'.join(test_example) + '\n'
        logger.debug(f"eval_example: {test_example}")
        accuracy = model.eval(test_data, example=test_example)
        logger.info(f"accuracy-test: {accuracy}")

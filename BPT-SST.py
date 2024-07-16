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
logger.write('BPT-SST.txt')


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
    parser.add_argument("--experiment_name", type=str, default='BPT-SST', help="实验名称")
    parser.add_argument("--seed", type=int, default=666, help="A seed for shuffle.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to run wandb.")
    parser.add_argument("--label_to_id", default={"Terrible": 0, "Great": 1}, type=ast.literal_eval)
    parser.add_argument("--train_data_path", type=str, default='data/SST/train.pt', help="The path of the train data.")
    parser.add_argument("--test_data_path", type=str, default='data/SST/test.pt', help="The path of the test data.")
    parser.add_argument("--vocabulary_data_path", type=str, default='data/SST/vocabulary.pt',
                        help="The path of the vocabulary data.")
    parser.add_argument("--llm_service", type=str, default='openai', help="The LLM service, openai, tongyi, zhipuai")
    parser.add_argument("--llm_key", type=str, default='lm-studio',
                        help="The LLM service api-key")
    parser.add_argument("--k_shot_train", default=10, type=int, help="每种类别参与训练的数据量")
    parser.add_argument("--k_shot_test", default=0, type=int, help="每种类别参与测试的数据量，0 denotes full-shot")
    parser.add_argument("--train_epochs", type=int, default=300, help="Total number of training epochs to perform.")
    parser.add_argument("--save_ckpt_path", type=str, default="ckpt/BPT/SST", help="The path of the checkpoint params.")
    parser.add_argument("--use_ckpt", type=bool, default=False, help="If True, will use checkpoint.")
    parser.add_argument("--use_ckpt_path", type=str, default="", help="The path of the checkpoint params.")
    parser.add_argument("--just_test", type=bool, default=False, help="If True, will just test.")
    parser.add_argument("--test_ckpt_path", type=str, default="ckpt/BPT/SST/",
                        help="The path of the checkpoint params.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="优化器学习率")
    parser.add_argument("--sample_size", type=int, default=5, help="求解策略梯度时的采样次数")
    parser.add_argument("--prompt_length", type=int, default=50, help="控制prompt生成长度(用词量)")
    parser.add_argument("--prompt_search_space", type=int, default=200, help="控制vocabulary搜索范围")
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
        ngram_list = vocabulary_loader(args.vocabulary_data_path)
        if not args.use_ckpt:
            prompts_probs = torch.FloatTensor(
                [[1 / args.prompt_search_space] * args.prompt_search_space] * args.prompt_length)
            prompts_probs.requires_grad = True
            optimizer = Adam([{
                "params": [prompts_probs],
                "weight_decay": args.weight_decay,  # 表示权重衰减，L2正则化项的系数，用于防止过拟合
                "lr": args.learning_rate
            }, ])
        else:
            if not os.path.exists(args.use_ckpt_path):
                logger.error(f"{args.use_ckpt_path} not exists")
            prompts_probs = torch.FloatTensor(probs_loader(f'{args.use_ckpt_path}/prompts_probs_best.pth').tolist())
            prompts_probs.requires_grad = True
            optimizer = Adam([{
                "params": [prompts_probs],
                "weight_decay": args.weight_decay,  # 表示权重衰减，L2正则化项的系数，用于防止过拟合
                "lr": args.learning_rate
            }, ])
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.train_epochs}")
        logger.info(f"  Train data size = {len(train_data['inputs'])}")
        logger.info(f"  Eval data size = {len(eval_data['inputs'])}")
        logger.info(f"  vocabulary size = {args.prompt_search_space}")
        logger.info(f"  Gradient Accumulation steps = {args.sample_size}")
        logger.info(f"  Length of prompt = {args.prompt_length}")
        loss_step = 0
        for epoch in range(args.train_epochs):
            prompts_dist = torch.distributions.Categorical(prompts_probs)
            prompts_discrete_indices_list = []
            loss_list = []
            for k in range(args.sample_size):
                # 进行一次采样
                prompts_discrete_indices = prompts_dist.sample()
                prompts_discrete_indices_list.append(prompts_discrete_indices)
                indices_list = prompts_discrete_indices.int().tolist()
                prompts_discrete_ngram_list = []
                for idx in indices_list:
                    prompts_discrete_ngram_list.append(ngram_list[idx])
                # 本次采样组成的提示语
                prompts_discrete = ' '.join(prompts_discrete_ngram_list)
                # 追加了采样后的prompt内容后形成的新prompt
                step_prompt = []
                for i in range(len(train_data['inputs'])):
                    step_prompt.append(train_data['inputs'][i].replace('[note]', f'\"{prompts_discrete}\"'))
                label_keys = list(args.label_to_id.keys())
                converted_target = torch.tensor([args.label_to_id[label] for label in train_data['labels']])
                pred, logits = model.get_pred_result(step_prompt)
                loss = ce_loss(logits.view(-1, len(label_keys)), converted_target)
                loss_list.append(loss.item())
            loss_avg = sum(loss_list) / args.sample_size  # 用一个batch中的所有样本计算loss
            logger.info(f"train_loss: {loss_avg}, epoch: {epoch}")
            if args.use_wandb:
                wandb.log({'train_loss': loss_avg, 'epoch': epoch})
            # 清零梯度，为优化变量手动计算梯度
            optimizer.zero_grad()
            # .repeat(1, 1, 1) 是对对应的每一个维度进行复制,(2,2,2),就是第一维乘2，第二维乘2，第三维乘2
            derivative = (-1 / (prompts_probs + 1e-6)).repeat(args.sample_size, 1, 1)
            # derivative梯度矩阵，为对角线上取倒数，非对角线上-1*倒数
            for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                for i in range(args.prompt_length):
                    derivative[k][i][prompts_discrete_indices[i]] *= -1
            # 梯度估计
            prompts_probs.grad = torch.zeros_like(prompts_probs)
            for k in range(args.sample_size):
                prompts_probs.grad += 1 / (args.sample_size - 1 + 1e-6) * (loss_list[k] - loss_avg) * derivative[k]
            # 完成一次参数更新
            optimizer.step()
            # 对参数进行原地relu
            constrainScoreByWholeExact(prompts_probs.detach())
            eval_prompt = []
            for idx in prompts_probs.argmax(1):
                eval_prompt.append(ngram_list[idx])
            eval_prompt = ' '.join(eval_prompt)
            accuracy = model.eval(eval_data, prompt=eval_prompt)
            if accuracy >= args.best_accuracy:
                args.best_accuracy = accuracy
                probs_path = os.path.join(args.save_ckpt_path, args.experiment_id)
                if not os.path.exists(probs_path):
                    os.makedirs(probs_path)
                torch.save(prompts_probs, f"{probs_path}/prompts_probs_best.pth")
                if args.use_wandb:
                    wandb.alert(title='best_eval', text=f'epoch: {epoch}, accuracy: {accuracy}')
            else:
                probs_path = os.path.join(args.save_ckpt_path, args.experiment_id)
                if not os.path.exists(probs_path):
                    os.makedirs(probs_path)
                torch.save(prompts_probs, f"{probs_path}/prompts_probs_latest.pth")
            logger.info(f"eval-accuracy: {accuracy}, epoch: {epoch}")
            if args.use_wandb:
                wandb.log({'eval-accuracy': accuracy, 'epoch': epoch})
    else:
        if not os.path.exists(args.test_ckpt_path):
            logger.error(f"{args.test_ckpt_path} not exists")
        prompts_probs = torch.FloatTensor(probs_loader(f'{args.test_ckpt_path}/prompts_probs_best.pth').tolist())
        test_data = data_loader(args.test_data_path, args.k_shot_test)
        ngram_list = vocabulary_loader(args.vocabulary_data_path)
        logger.info("***** Running testing *****")
        logger.info(f"  Test data size = {len(test_data['inputs'])}")
        # 对本次更新结果进行评估
        test_prompt = []
        logger.debug(f"prompts_probs_argmax: {prompts_probs.argmax(1)}")
        for idx in prompts_probs.argmax(1):
            test_prompt.append(ngram_list[idx])
        test_prompt = ' '.join(test_prompt)
        logger.debug(f"eval_prompt: {test_prompt}")
        accuracy = model.eval(test_data, prompt=test_prompt)
        logger.info(f"accuracy-test: {accuracy}")

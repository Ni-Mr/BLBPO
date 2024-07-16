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
logger.write('BLBPO-5G.txt')


def get_prompt(ent1, ent2, text):
    input_format = 'In 5G network, Whether [ent1] related to [ent2] ? Some contextual information: [text]. Note: [note]. Respond ONLY with "Yes" or "No". Answer: '
    prompt = 'Question: ' + input_format
    if '[ent1]' in prompt:
        prompt = prompt.replace('[ent1]', f'\"{ent1}\"')
    if '[ent2]' in prompt:
        prompt = prompt.replace('[ent2]', f'\"{ent2}\"')
    if '[text]' in prompt:
        prompt = prompt.replace('[text]', f'\"{text}\"')
    return prompt


def data_loader(filepath, k_shot=0):
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    data_ls = []
    for it in zip(loaded_data['ent1'], loaded_data['ent2'], loaded_data['text'], loaded_data['rel']):
        data_ls.append(it)
    random.shuffle(data_ls)
    data_item = []
    if k_shot:
        shot = k_shot
        for it in data_ls:
            if it[-1] == 'No':
                data_item.append(it)
                shot -= 1
            if shot == 0:
                break
        shot = k_shot
        for it in data_ls:
            if it[-1] == 'Yes':
                data_item.append(it)
                shot -= 1
            if shot == 0:
                break
    else:
        data_item = data_ls
    result = {'inputs': [], 'labels': []}
    for it in tqdm(data_item):
        result["inputs"].append(get_prompt(ent1=it[0], ent2=it[1], text=it[2]))
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
    parser.add_argument("--experiment_name", type=str, default='BLBPO-5G', help="实验名称")
    parser.add_argument("--seed", type=int, default=666, help="A seed for shuffle.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to run wandb.")
    parser.add_argument("--label_to_id", default={"No": 0, "Yes": 1}, type=ast.literal_eval)
    parser.add_argument("--train_data_path", type=str, default='data/5G/train.pt', help="The path of the train data.")
    parser.add_argument("--test_data_path", type=str, default='data/5G/test.pt', help="The path of the test data.")
    parser.add_argument("--example_data_path", type=str, default='data/5G/example.pt',
                        help="The path of the example data.")
    parser.add_argument("--vocabulary_data_path", type=str, default='data/5G/vocabulary.pt',
                        help="The path of the vocabulary data.")
    parser.add_argument("--llm_service", type=str, default='openai', help="The LLM service, openai, tongyi, zhipuai")
    parser.add_argument("--llm_key", type=str, default='lm-studio',
                        help="The LLM service api-key")
    parser.add_argument("--k_shot_train", default=10, type=int, help="每种类别参与训练的数据量")
    parser.add_argument("--k_shot_test", default=0, type=int, help="每种类别参与测试的数据量，0 denotes full-shot")
    parser.add_argument("--train_epochs", type=int, default=600, help="Total number of training epochs to perform.")
    parser.add_argument("--save_ckpt_path", type=str, default="ckpt/BLBPO/5G", help="The path of the checkpoint params.")
    parser.add_argument("--use_ckpt", type=bool, default=False, help="If True, will use checkpoint.")
    parser.add_argument("--use_ckpt_path", type=str, default="", help="The path of the checkpoint params.")
    parser.add_argument("--just_test", type=bool, default=False, help="If True, will just test.")
    parser.add_argument("--test_ckpt_path", type=str, default="ckpt/BLBPO/5G/",
                        help="The path of the checkpoint params.")
    parser.add_argument("--lr", type=float, default=0.001, help="割平面更新的学习率")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="优化器学习率")
    parser.add_argument("--k1_steps", type=int, default=2, help="对上层问题进行K次迭代")
    parser.add_argument("--sample_size", type=int, default=5, help="求解策略梯度时的采样次数")
    parser.add_argument("--k2_steps", type=int, default=2, help="对下层问题进行K次迭代")
    parser.add_argument("--example_number", type=int, default=2, help="每种类别被选择的样本的数量")
    parser.add_argument("--example_search_space", type=int, default=100, help="example的搜索空间")
    parser.add_argument("--prompt_length", type=int, default=50, help="控制prompt生成长度(用词量)")
    parser.add_argument("--prompt_search_space", type=int, default=100, help="控制vocabulary搜索范围")
    parser.add_argument("--P", default=[], type=list, help="割平面约束集合.")
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
        ngram_list = vocabulary_loader(args.vocabulary_data_path)
        lmd = [[0.0] * args.prompt_search_space] * args.prompt_length
        if not args.use_ckpt:
            example_probs = torch.FloatTensor(
                [[1 / args.example_search_space] * args.example_search_space] * args.example_number * 2)
            example_probs.requires_grad = True
            prompts_probs = torch.FloatTensor(
                [[1 / args.prompt_search_space] * args.prompt_search_space] * args.prompt_length)
            prompts_probs.requires_grad = True
            optimizer = Adam([{
                "params": [example_probs, prompts_probs],
                "weight_decay": args.weight_decay,  # 表示权重衰减，L2正则化项的系数，用于防止过拟合
                "lr": args.learning_rate
            }, ])
        else:
            if not os.path.exists(args.use_ckpt_path):
                logger.error(f"{args.use_ckpt_path} not exists")
            example_probs = torch.FloatTensor(probs_loader(f'{args.use_ckpt_path}/example_probs_best.pth').tolist())
            example_probs.requires_grad = True
            prompts_probs = torch.FloatTensor(probs_loader(f'{args.use_ckpt_path}/prompts_probs_best.pth').tolist())
            prompts_probs.requires_grad = True
            optimizer = Adam([{
                "params": [example_probs, prompts_probs],
                "weight_decay": args.weight_decay,  # 表示权重衰减，L2正则化项的系数，用于防止过拟合
                "lr": args.learning_rate
            }, ])
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.train_epochs}")
        logger.info(f"  Train data size = {len(train_data['inputs'])}")
        logger.info(f"  Eval data size = {len(eval_data['inputs'])}")
        logger.info(f"  Example data size = {args.example_search_space}")
        logger.info(f"  vocabulary size = {args.prompt_search_space}")
        logger.info(f"  Gradient Accumulation steps = {args.sample_size}")
        logger.info(f"  Iterations of the upper level problem = {args.k1_steps}")
        logger.info(f"  Iterations of the lower level problem = {args.k2_steps}")
        logger.info(f"  Number of example = {args.example_number * 2}")
        logger.info(f"  Length of prompt = {args.prompt_length}")
        loss_step = 0
        for epoch in range(args.train_epochs):
            # 对上层问题进行K次迭代
            for _ in range(args.k1_steps):
                example_dist = torch.distributions.Categorical(example_probs)
                examples_discrete_indices_list = []
                prompts_dist = torch.distributions.Categorical(prompts_probs)
                prompts_discrete_indices_list = []
                loss_list = []
                for k in range(args.sample_size):
                    # 进行一次采样
                    examples_discrete_indices = example_dist.sample()
                    examples_discrete_indices_list.append(examples_discrete_indices)
                    indices_list_sample = examples_discrete_indices.int().tolist()
                    examples_discrete_list = []
                    prompts_discrete_indices = prompts_dist.sample()
                    prompts_discrete_indices_list.append(prompts_discrete_indices)
                    indices_list = prompts_discrete_indices.int().tolist()
                    prompts_discrete_ngram_list = []
                    # 针对example进行抽取
                    for idx in range(2):
                        for idy in range(args.example_number):
                            index = indices_list_sample[idx * args.example_number + idy]
                            examples_discrete_list.append(example_data[list(args.label_to_id.keys())[idx]][index])
                    # 针对prompt进行抽取
                    for idx in indices_list:
                        prompts_discrete_ngram_list.append(ngram_list[idx])
                    # 本次采样组成的提示语
                    examples_discrete = '\n'.join(examples_discrete_list) + '\n'
                    prompts_discrete = ' '.join(prompts_discrete_ngram_list)
                    # 追加了采样后的内容后形成的新question
                    step_question = []
                    for i in range(len(train_data['inputs'])):
                        step_question.append(
                            examples_discrete.replace('[note]', f'\"{prompts_discrete}\"') + train_data['inputs'][
                                i].replace(
                                '[note]', f'\"{prompts_discrete}\"'))
                    logger.debug(f"step_question: {step_question[0]}")
                    converted_target = torch.tensor([args.label_to_id[label] for label in train_data['labels']])
                    pred, logits = model.get_pred_result(step_question)
                    loss = ce_loss(logits.view(-1, len(args.label_to_id)), converted_target)
                    loss_list.append(loss.item())
                loss_avg = sum(loss_list) / args.sample_size  # 对概率矩阵进行抽样取平均计算loss
                logger.info(f"train_loss: {loss_avg}, step: {loss_step}")
                if args.use_wandb:
                    wandb.log({'train_loss': loss_avg, 'step': loss_step})
                loss_step += 1
                # 清零梯度，为优化变量手动计算梯度
                optimizer.zero_grad()
                with torch.no_grad():
                    # 针对example参数直接计算梯度进行更新
                    derivative = (-1 / (example_probs + 1e-6)).repeat(args.sample_size, 1, 1)
                    for i, examples_discrete_indices in enumerate(examples_discrete_indices_list):
                        for j in range(2 * args.example_number):
                            derivative[i][j][examples_discrete_indices[j]] *= -1
                    example_probs.grad = torch.zeros_like(example_probs)
                    for i in range(args.sample_size):
                        example_probs.grad += 1 / (args.sample_size - 1 + 1e-6) * (loss_list[i] - loss_avg) * \
                                              derivative[i]
                    # 针对prompt参数计算梯度进行更新
                    # 计算第一个梯度
                    derivative = (-1 / (prompts_probs + 1e-6)).repeat(args.sample_size, 1, 1)
                    for i, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                        for j in range(args.prompt_length):
                            derivative[i][j][prompts_discrete_indices[j]] *= -1
                    prompts_probs.grad = torch.zeros_like(prompts_probs)
                    for i in range(args.sample_size):
                        prompts_probs.grad += 1 / (args.sample_size - 1 + 1e-6) * (loss_list[i] - loss_avg) * \
                                              derivative[i]
                    # 计算第二个梯度
                    for i in args.P:
                        prompts_probs.grad += i[0].detach() * i[1]
                    # 针对lamda参数进行梯度上升
                    for i in args.P:
                        i[0].grad = i[1] * prompts_probs.detach() + i[2]
                        i[0] += args.lr * i[0].grad
                        # 对lamda参数进行原地修正
                        i[0].detach().sub_(1e-6)
                        i[0].detach().relu_()
                        i[0].grad.zero_()
                # 完成example与prompt的参数更新
                torch.nn.utils.clip_grad_norm_([example_probs, prompts_probs], max_norm=100, norm_type=2)
                optimizer.step()
                # 对参数进行原地relu
                constrainScoreByWholeExact(example_probs.detach())
                constrainScoreByWholeExact(prompts_probs.detach())
            # 对本次更新结果进行评估
            eval_example = []
            eval_prompt = []
            for idx in range(2):
                for idy in range(args.example_number):
                    index = example_probs.argmax(1)[idx * args.example_number + idy]
                    eval_example.append(example_data[list(args.label_to_id.keys())[idx]][index])
            eval_example = examples_discrete = '\n'.join(eval_example) + '\n'
            logger.debug(f"prompts_probs_argmax: {prompts_probs.argmax(1)}")
            for idx in prompts_probs.argmax(1):
                eval_prompt.append(ngram_list[idx])
            eval_prompt = ' '.join(eval_prompt)
            logger.debug(f"eval_example: {eval_example}")
            logger.debug(f"eval_prompt: {eval_prompt}")
            accuracy = model.eval(eval_data, example=eval_example, prompt=eval_prompt)
            if accuracy >= args.best_accuracy:
                args.best_accuracy = accuracy
                probs_path = os.path.join(args.save_ckpt_path, args.experiment_id)
                if not os.path.exists(probs_path):
                    os.makedirs(probs_path)
                torch.save(example_probs, f"{probs_path}/example_probs_best.pth")
                torch.save(prompts_probs, f"{probs_path}/prompts_probs_best.pth")
                if args.use_wandb:
                    wandb.alert(title='best_eval', text=f'epoch: {epoch}, accuracy: {accuracy}')
            else:
                probs_path = os.path.join(args.save_ckpt_path, args.experiment_id)
                if not os.path.exists(probs_path):
                    os.makedirs(probs_path)
                torch.save(example_probs, f"{probs_path}/example_probs_latest.pth")
                torch.save(prompts_probs, f"{probs_path}/prompts_probs_latest.pth")
            logger.info(f"eval-accuracy: {accuracy}, epoch: {epoch}")
            if args.use_wandb:
                wandb.log({'eval-accuracy': accuracy, 'epoch': epoch})
            # 对下层问题进行K次迭代求解约束
            prompts_probs_k = torch.FloatTensor(prompts_probs.tolist())
            prompts_probs_k.requires_grad = True
            prompt_optimizer = Adam([{
                "params": [prompts_probs_k],
                "weight_decay": args.weight_decay,  # 表示权重衰减，L2正则化项的系数，用于防止过拟合
                "lr": args.learning_rate
            }, ])
            for _ in range(args.k2_steps):
                prompts_dist_k = torch.distributions.Categorical(prompts_probs_k)
                # 所有采样index
                prompts_discrete_indices_list_k = []
                loss_list_k = []
                for k in range(args.sample_size):
                    # 进行一次采样
                    prompts_discrete_indices_k = prompts_dist_k.sample()
                    prompts_discrete_indices_list_k.append(prompts_discrete_indices_k)
                    indices_list_k = prompts_discrete_indices_k.int().tolist()
                    prompts_discrete_ngram_list_k = []
                    for idx in indices_list_k:
                        prompts_discrete_ngram_list_k.append(ngram_list[idx])
                    # 本次采样组成的提示语
                    prompts_discrete_k = ' '.join(prompts_discrete_ngram_list_k)
                    # 追加了采样后的内容后形成的新question
                    step_question = []
                    for i in range(len(train_data['inputs'])):
                        step_question.append(
                            eval_example.replace('[note]', f'\"{prompts_discrete_k}\"') + train_data['inputs'][
                                i].replace(
                                '[note]', f'\"{prompts_discrete_k}\"'))
                    converted_target = torch.tensor([args.label_to_id[label] for label in train_data['labels']])
                    pred_k, logits_k = model.get_pred_result(step_question)
                    loss_k = ce_loss(logits_k.view(-1, len(args.label_to_id)), converted_target)
                    loss_list_k.append(loss_k.item())
                loss_avg_k = sum(loss_list_k) / args.sample_size
                prompt_optimizer.zero_grad()
                derivative = (-1 / (prompts_probs_k + 1e-6)).repeat(args.sample_size, 1, 1)
                for i, prompts_discrete_indices_k in enumerate(prompts_discrete_indices_list_k):
                    for j in range(args.prompt_length):
                        derivative[i][j][prompts_discrete_indices_k[j]] *= -1
                # 梯度估计
                prompts_probs_k.grad = torch.zeros_like(prompts_probs_k)
                for i in range(args.sample_size):
                    prompts_probs_k.grad += 1 / (args.sample_size - 1 + 1e-6) * (loss_list_k[i] - loss_avg_k) * \
                                            derivative[i]
                # 进行参数更新
                torch.nn.utils.clip_grad_norm_([prompts_probs_k], max_norm=100, norm_type=2)
                prompt_optimizer.step()
                constrainScoreByWholeExact(prompts_probs_k.detach())

            # 优化中层问题，求解上层问题的约束
            # L2 Norm or L1 Norm
            h_value = torch.norm(prompts_probs - prompts_probs_k)
            h_value.backward()
            tmp_0 = torch.FloatTensor(lmd)
            tmp_0.requires_grad = True
            tmp_1 = prompts_probs.grad
            tmp_2 = h_value - prompts_probs.grad * prompts_probs
            args.P.append([tmp_0, tmp_1, tmp_2])
            logger.debug(f"epochs: {epoch}, 平面数: {len(args.P)}")
    else:
        if not os.path.exists(args.test_ckpt_path):
            logger.error(f"{args.test_ckpt_path} not exists")
        example_probs = torch.FloatTensor(probs_loader(f'{args.test_ckpt_path}/example_probs_best.pth').tolist())
        prompts_probs = torch.FloatTensor(probs_loader(f'{args.test_ckpt_path}/prompts_probs_best.pth').tolist())
        test_data = data_loader(args.test_data_path, args.k_shot_test)
        example_data = example_loader(args.example_data_path)
        ngram_list = vocabulary_loader(args.vocabulary_data_path)
        logger.info("***** Running testing *****")
        logger.info(f"  Test data size = {len(test_data['inputs'])}")
        # 对本次更新结果进行评估
        test_example = []
        test_prompt = []
        for idx in range(2):
            for idy in range(args.example_number):
                index = example_probs.argmax(1)[idx * args.example_number + idy]
                test_example.append(example_data[list(args.label_to_id.keys())[idx]][index])
        test_example = examples_discrete = '\n'.join(test_example) + '\n'
        logger.debug(f"prompts_probs_argmax: {prompts_probs.argmax(1)}")
        for idx in prompts_probs.argmax(1):
            test_prompt.append(ngram_list[idx])
        test_prompt = ' '.join(test_prompt)
        logger.debug(f"eval_example: {test_example}")
        logger.debug(f"eval_prompt: {test_prompt}")
        accuracy = model.eval(test_data, example=test_example, prompt=test_prompt)
        logger.info(f"accuracy-test: {accuracy}")

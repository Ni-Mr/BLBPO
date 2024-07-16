import argparse
import os
import pickle
import ast
import torch
from utils.llm import LlmModel
from utils.log import Logger
import random
from tqdm import tqdm

logger = Logger(__name__)
logger.set_level(logger.INFO)


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
    parser.add_argument("--label_to_id", default={"No": 0, "Yes": 1}, type=ast.literal_eval)
    parser.add_argument("--test_data_path", type=str, default='data/5G/test.pt', help="The path of the test data.")
    parser.add_argument("--example_data_path", type=str, default='data/5G/example.pt',
                        help="The path of the example data.")
    parser.add_argument("--vocabulary_data_path", type=str, default='data/5G/vocabulary.pt',
                        help="The path of the vocabulary data.")
    parser.add_argument("--llm_service", type=str, default='tongyi', help="The LLM service, openai, tongyi, zhipuai")
    parser.add_argument("--llm_key", type=str, default='sk-fdfa0cbeb0be4ac78e44346f2e0c68e3',
                        help="The LLM service api-key")
    parser.add_argument("--k_shot_test", default=0, type=int, help="每种类别参与测试的数据量，0 denotes full-shot")
    parser.add_argument("--just_test", type=bool, default=True, help="If True, will just test.")
    parser.add_argument("--example_number", type=int, default=2, help="每种类别被选择的样本的数量")
    parser.add_argument("--test_ckpt_path", type=str, default="ckpt/5G/ffea85a5-8a74-4643-b714-ca6209cd3cce",
                        help="The path of the checkpoint params.")
    args = parser.parse_args()
    random.seed(args.seed)
    model = LlmModel(model_name=args.llm_service, api_key=args.llm_key)
    model.set_params(args.label_to_id)
    if args.just_test:
        if not os.path.exists(args.test_ckpt_path):
            logger.error(f"{args.test_ckpt_path} not exists")
        example_probs = torch.FloatTensor(probs_loader(f'{args.test_ckpt_path}/example_probs_latest.pth').tolist())
        prompts_probs = torch.FloatTensor(probs_loader(f'{args.test_ckpt_path}/prompts_probs_latest.pth').tolist())
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
        # print(test_example, test_prompt)
        accuracy = model.eval(test_data, example=test_example,prompt=test_prompt)
        logger.info(f"accuracy-test: {accuracy}")
        # accuracy = model.eval(test_data)
        # logger.info(f"accuracy-test: {accuracy}")

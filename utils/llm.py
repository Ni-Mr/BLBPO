import sys
import numpy as np
import torch
from langchain_community.chat_models import tongyi
from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI


class LlmModel:
    def __init__(self, model_name, api_key):
        self.label_to_id = None
        self.label_keys = None
        self.model = None
        self.model_name = model_name
        self.api_key = api_key
        self.create_model()

    def create_model(self):
        if self.model_name == "openai":
            self.model = ChatOpenAI(base_url="http://localhost:6006/v1", api_key=self.api_key)
        if self.model_name == "tongyi":
            self.model = tongyi.ChatTongyi(dashscope_api_key=self.api_key)
        if self.model_name == "zhipuai":
            self.model = ZhipuAI(api_key=self.api_key)

    def set_params(self, label_to_id):
        self.label_to_id = label_to_id
        self.label_keys = list(label_to_id.keys())

    def invoke(self, msg):
        response = None
        if self.model_name == "openai":
            response = self.model.invoke(input=msg, max_tokens=5, seed=666, temperature=0.1).content
        if self.model_name == "tongyi":
            response = self.model.invoke(input=msg, max_tokens=5, seed=666, model="qwen-turbo", ).content
        if self.model_name == "zhipuai":
            response = \
                self.model.chat.completions.create(model="glm-3-turbo", messages=[{"role": "user", "content": msg}],
                                                   seed=666, max_tokens=5, stream=False).choices[0].message.content
        # print(response)
        return response

    def get_pred_result(self, step_prompt):
        response = []
        try:
            for i in step_prompt:
                result = self.invoke(msg=i)
                response.append(result)
        except Exception as e:
            print(e)
            sys.exit("model invoke error")
        logits = self.get_logits(response, self.label_keys)
        pred = torch.argmax(logits, dim=-1)
        return pred, logits

    def get_logits(self, response, labels):
        if self.model_name == "openai" or self.model_name == "tongyi" or self.model_name == "zhipuai":
            logits = torch.zeros([len(response), len(labels)])
            for resi, ans in enumerate(response):
                for index, label in enumerate(labels):
                    if label in ans or ans in label:
                        # [0.8,1)范围内均匀分布的随机数
                        logits[resi, index] = np.random.uniform(0.8, 1)
            return logits

    def eval(self, eval_data, example='', prompt=''):
        response = []
        # 追加了采样后的内容后形成的新question
        step_question = []
        for i in range(len(eval_data['inputs'])):
            step_question.append(
                example.replace('[note]', f'\"{prompt}\"') + eval_data['inputs'][i].replace(
                    '[note]', f'\"{prompt}\"'))
        try:
            for i in step_question:
                print(i)
                result = self.invoke(msg=i)
                response.append(result)
                print(result)
        except Exception as e:
            print(e)
            sys.exit("model invoke error")
        # print(response)
        logits = self.get_logits(response, self.label_keys)
        pred = torch.argmax(logits, dim=-1)
        target = torch.tensor([self.label_to_id[label] for label in eval_data['labels']])
        # 计算预测是否正确
        correct = (pred == target).float()
        # 计算准确率
        accuracy = correct.sum() / len(target)
        accuracy = accuracy.item()
        # print("准确率:", accuracy)
        return accuracy


def constrainScoreByWholeExact(prompts_probs):
    # prompt_embeds: shape [prompt_len, emb_dim]
    # 既要保持概率分布，也要让和在0和1之间
    for i in range(len(prompts_probs)):
        if prompts_probs[i].min() < 0:
            prompts_probs[i].add_(-1 * prompts_probs[i].min())
        v, itr = solve_v_total_exact(prompts_probs[i])
        prompts_probs[i].sub_(v).clamp_(0, 1)


def solve_v_total_exact(prompt_emb):
    k = 1
    a, b = 0, 0
    b = prompt_emb.max()

    def f(_v):
        # .clamp(0, 1) 将取值裁剪到 0 和 1 之间
        s = (prompt_emb - _v).clamp(0, 1).sum()
        return s - k

    itr = 0
    v = 0
    while True:
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    return v, itr

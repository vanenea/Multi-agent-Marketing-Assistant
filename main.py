import abc
import os
import random
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from http import HTTPStatus
from dashscope import Application
load_dotenv()


# 基类 Agent
class Agent(abc.ABC):
    @abc.abstractmethod
    def run(self, context: dict) -> dict:
        pass


# 大模型调用封装 Agent 基类
class LLMAgent(Agent):
    def __init__(self, model_name: str = 'qwen-plus'):
        self.model_name = model_name

    def call_llm(self, prompt: str) -> str:
        chat = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=self.model_name,
            temperature=0.7,
            verbose=True
        )
        messages = [
            {"role": "system", "content": "You are a helpful marketing assistant."},
            {"role": "user", "content": prompt}
        ]
        resp = chat.invoke(messages)
        return resp.content


# 信息采集 Agent
class InformationCollectorAgent(Agent):
    def __init__(self, ali_token: str = None):
        self.ali_token = ali_token or os.getenv('DASHSCOPE_API_KEY')


    def run(self, context: dict) -> dict:
        keyword = context.get('product', '')
        response = Application.call(
            api_key=self.ali_token,
            app_id='ef8118d15105460297c03d8bffc3552f',  # 替换为实际的应用 ID
            prompt=f'根据用户提供的产品{keyword}, 然后抓取社交媒体、新闻、竞品网站等数据, 然后对数据做一个汇总')

        if response.status_code != HTTPStatus.OK:
            print(f'request_id={response.request_id}')
            print(f'code={response.status_code}')
            print(f'message={response.message}')
            print(f'请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code')
        else:
            #print(response.output.text)
            context['data'] = response.output.text
            return context


# 策略规划 Agent（基于大模型）
class StrategyPlannerAgent(LLMAgent):
    def run(self, context: dict) -> dict:
        goal = context.get('goal', '提高品牌曝光')
        budget = context.get('budget', '1000')
        prompt = (
            f"请根据以下信息制定营销策略：\n"
            f"目标：{goal}\n"
            f"预算：{budget} 元\n"
            f"产品：{context.get('product')}\n"
            f"请输出一个关键动作序列和时间节点。"
        )
        context['strategy'] = self.call_llm(prompt)
        return context


# 内容生成 Agent（基于大模型）
class ContentGeneratorAgent(LLMAgent):
    def run(self, context: dict) -> dict:
        product = context.get('product')
        audience = context.get('audience', '潜在用户')
        data = context.get('data')
        strategy = context.get('strategy')
        prompt = (
            f"信息采集，抓取社交媒体、新闻、竞品网站等数据，数据如下：{data}：\n"
            f"根据目标与预算，制定营销策略／关键动作序列，数据如下：{strategy}：\n"
            f"然后请为以下产品撰写推广文案，面向{audience}：\n"
            f"产品：{product}\n"
            f"要求：简洁、有吸引力，包含行动号召。"
        )
        context['content'] = self.call_llm(prompt)
        return context


# 投放执行 Agent
class ExecutionAgent(Agent):
    def run(self, context: dict) -> dict:
        # TODO: 与第三方平台对接，自动投放
        context['result'] = 'ad_posted'
        return context


# 多智能体营销助手
class MarketingAssistant:
    def __init__(self, excel_path: str, model_name: str = 'gpt-3.5-turbo'):
        #self.excel_path = excel_path
        self.agents = [
            InformationCollectorAgent(),
            StrategyPlannerAgent(model_name),
            ContentGeneratorAgent(model_name),
            ExecutionAgent(),
        ]

    def select_product(self) -> str:
        #df = pd.read_excel(self.excel_path)
        #return random.choice(df['产品名称'].tolist())
        return "小米手机15"

    def run(self, goal: str = None, budget: str = None, audience: str = None) -> dict:
        product = self.select_product()
        context = {'product': product}
        if goal:
            context['goal'] = goal
        if budget:
            context['budget'] = budget
        if audience:
            context['audience'] = audience

        for agent in self.agents:
            context = agent.run(context)

        return context


if __name__ == '__main__':
    model_name = 'qwen-plus-latest'
    assistant = MarketingAssistant('products.xlsx', model_name)
    result = assistant.run(goal='提高社交媒体关注度', budget='2000', audience='年轻人')
    print(result)
    #prompt ={'product': '小米手机15', 'goal': '提高社交媒体关注度', 'budget': '5000', 'audience': '年轻人'}

    # ifca = InformationCollectorAgent()
    # context=ifca.run(prompt)
    # print(context)

    # sga = StrategyPlannerAgent(model_name)
    # llm = sga.run(prompt)
    # print(llm)

    # cga = ContentGeneratorAgent()
    # cgc = cga.run(prompt)
    # print(cgc)
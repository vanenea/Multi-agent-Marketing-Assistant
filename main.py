import abc
import os
import random
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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
        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=[{'role': 'system', 'content': 'You are a helpful marketing assistant.'},
        #               {'role': 'user', 'content': prompt}],
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     api_key=os.getenv("DASHSCOPE_API_KEY"),
        #     max_tokens=300,
        #     temperature=0.7
        # )
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
    def __init__(self, weibo_token: str = None, xhs_token: str = None):
        self.weibo_token = weibo_token or os.getenv('WEIBO_ACCESS_TOKEN')
        self.xhs_token = xhs_token or os.getenv('XHS_ACCESS_TOKEN')

    def fetch_weibo(self, keyword: str, count: int = 10) -> list:
        """
        示例：调用微博搜索 API，返回最新微博列表
        """
        url = 'https://api.weibo.com/2/search/topics.json'
        params = {
            'access_token': self.weibo_token,
            'q': keyword,
            'count': count
        }
        resp = requests.get(url, params=params)
        items = []
        if resp.status_code == 200:
            data = resp.json()
            topics = data.get('topics', [])
            for t in topics:
                items.append({'source': 'weibo', 'text': t.get('trend_name'), 'timestamp': t.get('trend_time')})
        return items

    def fetch_xiaohongshu(self, keyword: str, count: int = 10) -> list:
        """
        示例：调用小红书搜索 API，返回笔记列表
        """
        url = 'https://www.xiaohongshu.com/fe_api/burdock/weixin/v2/search/notes'
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {
            'keyword': keyword,
            'page_size': count,
            'page': 1
        }
        if self.xhs_token:
            headers['Authorization'] = f'Bearer {self.xhs_token}'
        resp = requests.get(url, headers=headers, params=params)
        items = []
        if resp.status_code == 200:
            notes = resp.json().get('data', [])
            for n in notes:
                items.append({'source': 'xiaohongshu', 'text': n.get('desc'), 'timestamp': n.get('time')})
        return items

    def run(self, context: dict) -> dict:
        keyword = context.get('product', '')
        # 从微博和小红书抓取数据
        weibo_data = self.fetch_weibo(keyword, count=20)
        xhs_data = self.fetch_xiaohongshu(keyword, count=20)
        # 合并与时间排序
        combined = weibo_data + xhs_data
        combined.sort(key=lambda x: x['timestamp'], reverse=True)
        context['data'] = combined
        # 延迟以防 API 限制
        time.sleep(1)
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
        prompt = (
            f"请为以下产品撰写推广文案，面向{audience}：\n"
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
        self.excel_path = excel_path
        self.agents = [
            InformationCollectorAgent(),
            StrategyPlannerAgent(model_name),
            ContentGeneratorAgent(model_name),
            ExecutionAgent(),
        ]

    def select_product(self) -> str:
        df = pd.read_excel(self.excel_path)
        return random.choice(df['产品名称'].tolist())

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
    # assistant = MarketingAssistant('products.xlsx', model_name='gpt-4')
    # result = assistant.run(goal='提高社交媒体关注度', budget='2000', audience='年轻人')
    # print(result)
    ifca = InformationCollectorAgent()
    ifca.run()
    cga = ContentGeneratorAgent()
    llm = cga.run({'product': '手机', 'goal': '提高社交媒体关注度', 'budget': '2000', 'audience': '年轻人'})
    print(llm)
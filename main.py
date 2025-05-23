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
            f"要求：简洁、有吸引力，包含行动号召，直接给文案"
        )
        context['content'] = self.call_llm(prompt)
        return context


# 投放执行 Agent
class ExecutionAgent(Agent):
    def __init__(self):
        # 从环境变量读取各平台凭证
        self.weibo_token = os.getenv("WEIBO_ACCESS_TOKEN")
        self.xhs_token   = os.getenv("XHS_ACCESS_TOKEN")
        if not self.weibo_token or not self.xhs_token:
            raise ValueError("请设置 WEIBO_ACCESS_TOKEN 和 XHS_ACCESS_TOKEN 环境变量")

        # API endpoints（示例，实际请按文档填写）
        self.weibo_post_url = "https://api.weibo.com/2/statuses/share.json"
        self.xhs_post_url   = "https://open.xiaohongshu.com/api/v1/posts"

    def post_to_weibo(self, text: str, image_urls: list[str] = None) -> dict[str, any]:
        """
        调用微博分享接口发布一条纯文本或图文微博。
        - text: 发布的文字内容
        - image_urls: 已上传到微博服务器的图片 URL 列表（可选）
        """
        payload = {
            "access_token": self.weibo_token,
            "status": text,
        }
        if image_urls:
            # 如果要发布图文，将 image_urls 合并成逗号分隔
            payload["pic_ids"] = ",".join(image_urls)

        resp = requests.post(self.weibo_post_url, data=payload)
        result = resp.json()
        if resp.status_code != 200 or "error" in result:
            return {"status": "failed", "raw_response": result}

        # 成功返回包含 id 字段
        return {"status": "success", "post_id": str(result.get("id")), "raw_response": result}

    def post_to_xiaohongshu(self, title: str, body: str, cover_image: str = None) -> dict[str, any]:
        """
        调用小红书笔记发布接口。
        - title: 笔记标题
        - body: 笔记正文（支持 markdown 或 HTML）
        - cover_image: 笔记封面图 URL（可选）
        """
        headers = {
            "Authorization": f"Bearer {self.xhs_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "title": title,
            "content": body,
            # 可根据文档增加更多字段，比如话题、标签、商品链接等
        }
        if cover_image:
            payload["cover_image_url"] = cover_image

        resp = requests.post(self.xhs_post_url, json=payload, headers=headers)
        result = resp.json()
        if resp.status_code not in (200, 201) or result.get("code") != 0:
            return {"status": "failed", "raw_response": result}

        # 假设 result['data']['post_id'] 为新笔记的 ID
        post_id = result.get("data", {}).get("post_id")
        return {"status": "success", "post_id": post_id, "raw_response": result}

    def run(self, context: dict) -> dict:
        """
        执行投放。根据 context 中的 strategy 决定投放渠道和内容格式。
        需要在 context 中预先准备：
          - context['content']: dict, 如 {
              "weibo_text": "...",
              "weibo_images": [...],
              "xhs_title": "...",
              "xhs_body": "...",
              "xhs_cover": "..."
            }
          - context['strategy']: dict, 如 {"channels": ["weibo", "xiaohongshu"]}
        """
        results = []
        content  = context.get("content", {})
        strategy = context.get("strategy", {})

        channels = strategy.get("channels", [])
        for ch in channels:
            if ch.lower() == "weibo":
                text   = content.get("weibo_text", "")
                images = content.get("weibo_images", [])
                res = self.post_to_weibo(text=text, image_urls=images)
                res["channel"] = "weibo"
                results.append(res)

            elif ch.lower() in ("xiaohongshu", "xhs"):
                title = content.get("xhs_title", "")
                body  = content.get("xhs_body", "")
                cover = content.get("xhs_cover", None)
                res = self.post_to_xiaohongshu(title=title, body=body, cover_image=cover)
                res["channel"] = "xiaohongshu"
                results.append(res)

            else:
                results.append({
                    "channel": ch,
                    "status": "skipped",
                    "raw_response": f"不支持的渠道：{ch}"
                })

        # 将投放结果写回 context
        context["result"] = results
        return context


# 多智能体营销助手
class MarketingAssistant:
    def __init__(self, excel_path: str, model_name: str = 'qwen-plus-latest'):
        #self.excel_path = excel_path
        self.agents = [
            InformationCollectorAgent(),
            StrategyPlannerAgent(model_name),
            ContentGeneratorAgent(model_name),
            #ExecutionAgent(),
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
    result = assistant.run(goal='提高社交媒体关注度', budget='5000', audience='年轻人')
    print(result.get('content'))
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
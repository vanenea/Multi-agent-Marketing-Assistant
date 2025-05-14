from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()
# 初始化 LLM，base_url 指向兼容模式入口
chat = ChatOpenAI(
    api_key   = os.getenv("DASHSCOPE_API_KEY"),
    base_url  = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model     = "qwen-plus",           # 可替换为 qwen-max 等
    temperature = 0.7,
    verbose=True
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "介绍一下阿里云百炼。"}
]
resp = chat.invoke(messages)
print(resp.model_dump_json())
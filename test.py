import os

import requests
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_openai import ChatOpenAI

load_dotenv()


def llm():
    # 初始化 LLM，base_url 指向兼容模式入口
    chat = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",  # 可替换为 qwen-max 等
        temperature=0.7,
        verbose=True
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "介绍一下阿里云百炼。"}
    ]
    resp = chat.invoke(messages)
    print(resp.model_dump_json())


def firecrawl():
    # Initialize the FirecrawlApp with your API key
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    # Scrape a single URL
    url = 'https://www.zhihu.com/search?type=content&q=python%E6%95%99%E5%AD%A6'
    scraped_data = app.scrape_url(url,
                                  formats=['markdown'],
                                  headers={
                                      'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
                                      'Cookie': 'q_c1=86f37fd8be344ba5b38862201749e44c|1713518175000|1713518175000; _xsrf=IgBOO2DWHueidCqBgqApzknVECYBrXVP; edu_user_uuid=edu-v1|6e7ca80f-80a7-4693-b3e5-148d407658d8; z_c0=2|1:0|10:1746583550|4:z_c0|80:MS4xT3VZQ0VnQUFBQUFtQUFBQVlBSlZUZjROQ0dtc2Q1MjdydEVJVVJFSHVtaXhtVnVDQk5sQVFBPT0=|42912976a371adf10fef9b70dc6152f712963f18861c07c4eeba2c855fc4c4ed; _zap=a65803cd-5539-4138-a9cc-1db7c8d8baa9; d_c0=f7EThCrqcBqPTsX1mzEL6UWZ6l1syCdG9do=|1747034176; BEC=6bca8f185b99e85d761c7a0d8d692864; tst=r; SESSIONID=USKWYhwyliL9SQktzl5xCLQ33oowmjI6mp2Stj1eXg9; JOID=W1EcAkx8DD5Qd3Laa3Sy4r9GoblzKV99ASAarRoeZXJhIi2xH0KgFTB5ctJldXvs1ZtyTvh129nN09Rp3qTu3Qk=; osd=V1AWAkhwDTRQc37bYXS27r5Mob1_KFV9BSwbpxoaaXNrIim9HkigETx4eNJheXrm1Z9-T_J139XM2dRt0qXk3Q0=; __zse_ck=004_eRtYo3E9fZAvDD9oFPp6aWDb3RGBnjqe8VyAwej5t7yYGIH=orKC1Lgem1pOErb/RB6ReuwDjIQ/z6U6XmLQXANWj3hjDySlPqlbb7vUmXogMsrDdGitXINC/DcnfAal-QNkCX7/ue56UzuTdGJPl+UqCDFSWdmjJ0WftOpP9qN7TAyuNBQMQR1CeF14Trdpe1BdlRBGfO9vgaUyVNX77RUOwbh1kFHItyavu8MG6n8qwNH0ToYXvXW2n+nIpNZJy'
                                  },
                                  includeTags=[], excludeTags=[])
    print(scraped_data)


def request():
    res = requests.get("https://www.bing.com", verify=False)
    print(res.text)

if __name__ == '__main__':
    request()

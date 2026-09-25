import asyncio
import os
import time

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 读取.env配置文件中的信息，相关环境变量以.env文件中优先
load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 模型初始化
llm_deepseek = ChatDeepSeek(
    model="deepseek-flash",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL
)

async def demo_async_stream():
    print("=== ainvoke 的异步调用效果 ===")
    start_time = time.perf_counter()

    print("程序开始...")

    # 1. 创建任务 Task
    print(">>> 发起异步模型调用 astream...")
    stream_resp = llm_deepseek.astream("用一句话解释人工智能")

    # 2. 并行执行其他任务
    print(">>> 模型请求已在后台发送，继续执行其他异步任务...")
    for i in range(3):
        await asyncio.sleep(1)  # 使用异步等待，释放控制权
        print(f">>> 正在执行第{i + 1}个任务，（已耗时{time.perf_counter() - start_time:.2f}s）")

    # 3. 获取模型结果
    print(">>> 模型任务已完成，开始读取缓冲区中的流式结果")
    end_time = time.perf_counter()
    async for chunk in stream_resp:
        # LangChain消息块通过 .content 获取内容
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
    print(f"\n>>> 流式输出结果\n")
    print(f"=== 总运行时间：{end_time - start_time:.2f}s ===")

async def main():
    await demo_async_stream()

if __name__ == "__main__":
    asyncio.run(main())
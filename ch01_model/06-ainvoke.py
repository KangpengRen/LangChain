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

async def demo_async_invoke():
    print("=== ainvoke 的异步调用效果 ===")
    start_time = time.perf_counter()

    print("程序开始...")

    # 1. 创建任务 Task
    print(">>> 发起异步模型调用 ainvoke...")
    async_task = asyncio.create_task(llm_deepseek.ainvoke("用一句话解释人工智能"))

    # 2. 并行执行其他任务
    print(">>> 模型请求已在后台发送，继续执行本地逻辑...")
    for i in range(3):
        await asyncio.sleep(1)  # 使用异步等待，释放控制权
        print(f">>> 正在执行第{i + 1}个任务，（已耗时{time.perf_counter() - start_time:.2f}s）")

    # 3. 获取模型结果
    print(">>> 本地任务完成，检查模型状态...")
    response = await async_task

    end_time = time.perf_counter()
    print(f">>> 模型返回：{response.content}")
    print(f"=== 总运行时间：{end_time - start_time:.2f}s ===")

async def main():
    await demo_async_invoke()

if __name__ == "__main__":
    asyncio.run(main())
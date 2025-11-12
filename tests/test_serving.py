import asyncio
import os
from one_eval.serving.llm_client import get_llm_client
from one_eval.logger import get_logger

log = get_logger("test_serving")

async def test_api_client():
    """测试API客户端"""
    log.info("=== 测试API客户端 ===")
    
    # 从环境变量获取API密钥
    api_key = os.getenv("OE_API_KEY")
    if not api_key:
        log.error("请设置环境变量 OE_API_KEY")
        return
    
    try:
        # 创建API客户端
        client = get_llm_client(
            "api", 
            base_url="http://123.129.219.111:3000/v1/chat/completions", 
            api_key=api_key, 
            model="gpt-4o"
        )
        
        # 测试消息
        messages = [
            {"role": "user", "content": "你好,api是什么?"}
        ]
        
        # 调用API
        response = await client.achat(messages)
        log.info(f"API响应: {response}")
        
    except Exception as e:
        log.error(f"API客户端测试失败: {e}")

async def test_vllm_client():
    """测试vLLM本地客户端"""
    log.info("=== 测试vLLM本地客户端 ===")
    
    try:
        # 创建vLLM本地客户端（需要根据实际情况修改model_path）
        client = get_llm_client(
            "vllm", 
            model_path="/mnt/DataFlow/scy/Model/Qwen2.5-7B-Instruct",  # 可以替换为本地模型路径
            tensor_parallel_size=1
        )
        
        # 测试消息
        messages = [
            {"role": "user", "content": "你好,vllm是什么?"}
        ]
        
        # 调用模型
        response = await client.achat(messages)
        log.info(f"vLLM响应: {response}")
        
    except Exception as e:
        log.error(f"vLLM客户端测试失败: {e}")

async def main():
    """主函数"""
    log.info("开始测试llm_client")
    
    # 测试API客户端
    await test_api_client()
    
    # 测试vLLM本地客户端
    await test_vllm_client()
    
    log.info("测试完成")

if __name__ == "__main__":
    asyncio.run(main())
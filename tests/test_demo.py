import os
import asyncio
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from one_eval.core.state import NodeState
from one_eval.core.agent import CustomAgent
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.logger import get_logger

log = get_logger("test_serving")


# -----------------------------
# 测试用工具
# -----------------------------
@tool
def add_numbers(a: int, b: int) -> int:
    """计算两数之和"""
    return a + b


def build_test_state() -> NodeState:
    return NodeState()


# -----------------------------
# 测试 - 完整执行两轮，使模型输出最终答案
# -----------------------------
async def test_agent_with_tool():
    log.info("=== 测试 CustomLLMCaller + 工具绑定 ===")

    state = build_test_state()
    tm = get_tool_manager()

    tm.register_post_tool(add_numbers, role="CustomAgent")

    agent = CustomAgent(
        tool_manager=tm,
        model_name="gpt-4o",
        react_mode=False,
    )

    # 初始消息
    messages = [
        HumanMessage(content="请调用工具 add_numbers, 计算 13 + 29, 并只返回最后整数。")
    ]

    llm = agent.create_llm(state)

    # -----------------------------
    # 第1轮：模型提出 tool_call
    # -----------------------------
    response1 = await llm.call(messages, bind_post_tools=True)
    print("\n=== 第1轮模型输出（tool_call） ===")
    print(response1)

    tool_calls = response1.additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        print("模型未调用工具，终止。")
        return

    call = tool_calls[0]
    tool_id = call["id"]
    tool_name = call["function"]["name"]
    tool_args = json.loads(call["function"]["arguments"])

    # 执行工具
    tool_fn = tm.get_post_tools(agent.role_name)[0] #暂时只用一个工具
    print(f"调用工具: {tool_fn}")
    tool_result = tool_fn.invoke(tool_args)

    print(f"\n=== 工具执行结果: {tool_result} ===")

    # -----------------------------
    # 第2轮：把工具执行结果发回模型
    # -----------------------------
    follow_messages = [
        *messages,
        response1,  # 第一轮模型的 tool_call
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_id,
        ),
    ]

    response2 = await llm.call(follow_messages, bind_post_tools=False)

    print("\n=== 第2轮模型最终输出(正确答案) ===")
    print(response2)
    print("\n最终答案内容:", response2.content)


if __name__ == "__main__":
    asyncio.run(test_agent_with_tool())

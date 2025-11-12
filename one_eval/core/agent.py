from __future__ import annotations
from typing import Any
from dataflow_agent.agentroles.base_agent import BaseAgent
from one_eval.state import NodeState
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class CustomAgent(BaseAgent):
    """
    CustomAgent
    ----------
    继承自 DataFlow-Agent 的 BaseAgent
    在 one_eval 框架中用于执行单个节点逻辑。

    处理：
        - 执行前置工具
        - 调用 LLM
        - 更新状态结果
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ======= 必须实现的抽象属性 =======
    @property
    def role_name(self) -> str:
        return "CustomAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_default"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_default"

    # ======= 核心执行逻辑 =======
    async def run(self, state: NodeState, **kwargs) -> NodeState:
        """
        CustomAgent 核心执行函数：
        调用 BaseAgent.execute()，自动处理
        - pre_tool
        - LLM 调用
        - 结果解析 & 更新
        """
        log.info(f"[CustomAgent] 开始执行，state keys: {list(state.keys())}")
        updated_state = await self.execute(state, use_agent=False, **kwargs)
        log.info("[CustomAgent] 执行完成")
        return updated_state

    # ======= 可选：自定义解析逻辑 =======
    def parse_result(self, content: str) -> dict[str, Any]:
        """可重写解析逻辑(默认复用 BaseAgent 的 robust_parse_json)"""
        result = super().parse_result(content)
        if "raw" in result:
            log.warning("[CustomAgent] LLM 输出未解析为 JSON, 使用 raw 内容。")
        return result

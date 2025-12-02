from __future__ import annotations
from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.agents.bench_search_agent import BenchSearchAgent
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.logger import get_logger

log = get_logger("BenchSearchNode")


class BenchSearchNode(BaseNode):
    """Step 2 Node：Benchmark 搜索节点"""

    def __init__(self):
        self.name = "BenchSearchNode"

    async def run(self, state: NodeState) -> NodeState:
        log.info(f"[{self.name}] 节点开始执行")

        tm = get_tool_manager()

        # 创建 Agent
        agent = BenchSearchAgent(
            tool_manager=tm,
            model_name="gpt-4o",
        )

        new_state = await agent.run(state)

        # Agent 会更新 state.agent_results["bench_search"]
        log.info(f"[{self.name}] 执行结束，bench_search 结果数量: "
                 f"{len(new_state.agent_results.get('bench_search', []))}")

        return new_state

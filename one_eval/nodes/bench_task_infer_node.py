from __future__ import annotations

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.agents.bench_task_infer_agent import BenchTaskInferAgent
from one_eval.logger import get_logger

log = get_logger("BenchTaskInferNode")


class BenchTaskInferNode(BaseNode):
    """
    Step3-Node2: BenchTaskInferNode
    - 包装 BenchTaskInferAgent
    - 对每个 bench 进行任务类型判定和字段映射
    """

    def __init__(self):
        self.name = "BenchTaskInferNode"
        self.agent = BenchTaskInferAgent()

    async def run(self, state: NodeState) -> NodeState:
        state.current_node = self.name
        
        # 委托给 Agent 执行
        new_state = await self.agent.run(state)
        
        return new_state

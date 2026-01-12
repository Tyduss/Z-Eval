from __future__ import annotations

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.agents.bench_config_recommend_agent import BenchConfigRecommendAgent
from one_eval.logger import get_logger

log = get_logger("BenchConfigRecommendNode")


class BenchConfigRecommendNode(BaseNode):
    """
    Step2-Node2: BenchConfigRecommendNode
    - 简单的 Wrapper 节点，负责调用 BenchConfigRecommendAgent
    """

    def __init__(self):
        self.name = "BenchConfigRecommendNode"
        self.agent = BenchConfigRecommendAgent()

    async def run(self, state: NodeState) -> NodeState:
        state.current_node = self.name
        
        # 委托给 Agent 执行
        new_state = await self.agent.run(state)
        
        return new_state

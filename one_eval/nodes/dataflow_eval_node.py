from __future__ import annotations

import os
from pathlib import Path

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState, ModelConfig
from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool
from one_eval.logger import get_logger

log = get_logger("DataFlowEvalNode")


class DataFlowEvalNode(BaseNode):
    """
    Step4: DataFlowEvalNode
    - 遍历 benches
    - 检查 eval_status
    - 准备 ModelConfig
    - 调用 DataFlowEvalTool
    - 更新状态
    """

    def __init__(self):
        self.name = "DataFlowEvalNode"
        self.logger = log
        # 默认输出目录
        self.output_root = os.path.join(os.getcwd(), "cache", "eval_results")

    async def run(self, state: NodeState) -> NodeState:
        state.current_node = self.name

        benches = getattr(state, "benches", None)
        if not benches:
            self.logger.warning("[DataFlowEvalNode] state.benches 为空")
            return state

        # 1. 准备模型配置
        # 优先使用 state.target_model，如果没有则尝试从 target_model 推断
        model_config = state.target_model
        
        if not model_config:
            # 简单的自动配置兜底
            if state.target_model:
                self.logger.info(f"Using default config for model: {state.target_model}")
                model_config = ModelConfig(
                    model_name_or_path=state.target_model,
                    is_api=False, # 默认为本地模型
                    tensor_parallel_size=1,
                    max_tokens=2048
                )
            else:
                self.logger.error("No target_model or target_model found!")
                return state

        tool = DataFlowEvalTool(output_root=self.output_root)

        for bench in benches:
            # 2. 检查是否跳过
            if bench.eval_status == "success":
                # 检查结果文件是否存在
                if bench.meta and bench.meta.get("eval_result"):
                    self.logger.info(f"[{bench.bench_name}] 已评测成功，跳过")
                    continue
            
            # 检查是否有必要的前置信息
            if not bench.dataset_cache:
                self.logger.warning(f"[{bench.bench_name}] 缺少 dataset_cache，跳过")
                bench.eval_status = "failed"
                continue

            if not bench.bench_dataflow_eval_type:
                self.logger.warning(f"[{bench.bench_name}] 缺少 eval_type，跳过")
                bench.eval_status = "failed"
                continue
            
            try:
                self.logger.info(f"[{bench.bench_name}] 开始评测... Type={bench.bench_dataflow_eval_type}")
                bench.eval_status = "running"
                
                # 3. 执行评测
                result = tool.run_eval(bench, model_config)
                
                # 4. 更新状态
                if not bench.meta:
                    bench.meta = {}
                
                bench.meta["eval_result"] = result["stats"]
                bench.meta["eval_detail_path"] = result["detail_path"]
                bench.eval_status = "success"
                
                self.logger.info(f"[{bench.bench_name}] 评测完成。Stats: {result['stats']}")

            except Exception as e:
                self.logger.error(f"[{bench.bench_name}] 评测失败: {e}")
                bench.eval_status = "failed"
                if not bench.meta:
                    bench.meta = {}
                bench.meta["eval_error"] = str(e)

        return state

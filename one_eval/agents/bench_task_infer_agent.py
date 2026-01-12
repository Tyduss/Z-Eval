from __future__ import annotations

import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("BenchTaskInferAgent")


class BenchTaskInferAgent(CustomAgent):
    @property
    def role_name(self) -> str:
        return "BenchTaskInferAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "bench_task_infer.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "bench_task_infer.task"

    async def run(self, state: NodeState) -> NodeState:
        # Agent 不更新 current_node

        benches = getattr(state, "benches", None)
        if not benches:
            return state

        llm = self.create_llm(state)

        for bench in benches:
            # 如果已经有 eval_type 和 mapping，跳过
            if bench.bench_dataflow_eval_type and bench.meta.get("key_mapping"):
                log.info(f"[{bench.bench_name}] 跳过判定，已存在 eval_type: {bench.bench_dataflow_eval_type}")
                continue

            if not bench.bench_keys:
                log.warning(f"[{bench.bench_name}] 跳过判定，无 Keys 信息")
                continue

            msgs = [
                SystemMessage(content=self.get_prompt(self.system_prompt_template_name)),
                HumanMessage(
                    content=self.get_prompt(
                        self.task_prompt_template_name,
                        bench_name=bench.bench_name,
                        keys=json.dumps(bench.bench_keys, ensure_ascii=False),
                    )
                ),
            ]

            try:
                resp = await llm.ainvoke(msgs)
                content = resp.content

                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                result = json.loads(content)
                
                eval_type = result.get("eval_type")
                key_mapping = result.get("key_mapping")

                if eval_type and key_mapping:
                    bench.bench_dataflow_eval_type = eval_type
                    if not bench.meta:
                        bench.meta = {}
                    bench.meta["key_mapping"] = key_mapping
                    log.info(f"[{bench.bench_name}] 判定结果: {eval_type}, Mapping: {key_mapping}")
                else:
                    log.warning(f"[{bench.bench_name}] LLM 返回格式不完整: {content}")

            except Exception as e:
                log.error(f"[{bench.bench_name}] 任务判定失败: {e}")

        return state

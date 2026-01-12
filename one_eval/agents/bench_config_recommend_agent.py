from __future__ import annotations

import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("BenchConfigRecommendAgent")


class BenchConfigRecommendAgent(CustomAgent):
    @property
    def role_name(self) -> str:
        return "BenchConfigRecommendAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "bench_config_recommend.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "bench_config_recommend.task"

    async def run(self, state: NodeState) -> NodeState:
        # Agent 不是 Node，没有 self.name 属性，直接使用 role_name 或者不设置 current_node
        # state.current_node 通常由 Node 层设置
        # state.current_node = self.role_name 


        benches = getattr(state, "benches", None)
        if not benches:
            return state

        # 创建 LLM Caller
        llm = self.create_llm(state)

        for bench in benches:
            if not bench.meta:
                continue
            
            # 如果已经有推荐配置，跳过
            if bench.meta.get("download_config"):
                log.info(f"[{bench.bench_name}] 跳过推荐，已存在 download_config")
                continue

            structure = bench.meta.get("structure")

            if not structure or not structure.get("ok"):
                log.warning(f"跳过 {bench.bench_name}: 无结构信息")
                continue

            repo_id = structure.get("repo_id") or bench.bench_name

            # 构造 Prompt
            # 为了节省 Token，只提取关键信息
            simplified_structure = {"subsets": []}
            for subset in structure.get("subsets", []):
                s_info = {
                    "subset": subset.get("subset"),
                    "splits": [s.get("name") for s in subset.get("splits", [])],
                }
                simplified_structure["subsets"].append(s_info)

            msgs = [
                SystemMessage(content=self.get_prompt(self.system_prompt_template_name)),
                HumanMessage(
                    content=self.get_prompt(
                        self.task_prompt_template_name,
                        repo_id=repo_id,
                        structure_json=json.dumps(
                            simplified_structure, ensure_ascii=False, indent=2
                        ),
                    )
                ),
            ]

            try:
                resp = await llm.ainvoke(msgs)
                content = resp.content

                # 解析 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                recommendation = json.loads(content)

                # 校验
                if "config" in recommendation and "split" in recommendation:
                    bench.meta["download_config"] = recommendation
                    log.info(f"[{repo_id}] 推荐配置: {recommendation}")
                else:
                    log.warning(f"[{repo_id}] LLM 返回格式错误: {content}")

            except Exception as e:
                log.error(f"[{repo_id}] 推荐失败: {e}")

        return state

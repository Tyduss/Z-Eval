from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional, List

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState, ModelConfig
from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool
from one_eval.logger import get_logger
from langgraph.types import Command
from one_eval.runtime.progress_store import set_progress, clear_progress

log = get_logger("DataFlowEvalNode")
VALID_EVAL_TYPES = {
    "key1_text_score",
    "key2_qa",
    "key2_q_ma",
    "key3_q_choices_a",
    "key3_q_choices_as",
    "key3_q_a_rejected",
}


class DataFlowEvalNode(BaseNode):
    """
    Step4: DataFlowEvalNode
    - 遍历 benches
    - 检查 eval_status
    - 准备 ModelConfig（支持多模型）
    - 调用 DataFlowEvalTool（并行执行多模型）
    - 更新状态
    """

    def __init__(self):
        self.name = "DataFlowEvalNode"
        self.logger = log
        # 默认输出目录
        self.output_root = os.path.join(os.getcwd(), "cache", "eval_results")
        self._tool = DataFlowEvalTool(output_root=self.output_root)

    def _get_model_configs(self, state: NodeState) -> List[ModelConfig]:
        """获取模型配置列表，优先使用 target_models，否则回退到 target_model"""
        models = getattr(state, "target_models", None) or []
        if models:
            return models

        # 单模型回退
        if state.target_model:
            return [state.target_model]

        # 兜底：从 target_model_name 推断
        if state.target_model_name:
            self.logger.info(f"Using default config for model: {state.target_model_name}")
            return [ModelConfig(
                model_name_or_path=state.target_model_name,
                is_api=False,
                tensor_parallel_size=1,
                max_tokens=2048
            )]

        return []

    async def run(self, state: NodeState, config: Optional[Any] = None) -> NodeState:
        state.current_node = self.name
        thread_id = None
        try:
            if isinstance(config, dict):
                thread_id = ((config.get("configurable") or {}).get("thread_id"))
        except Exception:
            thread_id = None

        benches = getattr(state, "benches", None)
        if not benches:
            self.logger.warning("[DataFlowEvalNode] state.benches 为空")
            return state

        # 1. 获取模型配置列表
        model_configs = self._get_model_configs(state)
        if not model_configs:
            self.logger.error("No target_model(s) found!")
            return state

        is_multi_model = len(model_configs) > 1
        if is_multi_model:
            self.logger.info(f"Multi-model evaluation: {[m.model_name_or_path for m in model_configs]}")

        tool = self._tool
        cursor = int(getattr(state, "eval_cursor", 0) or 0)
        if cursor < 0:
            cursor = 0
        if cursor >= len(benches):
            return state

        bench = benches[cursor]

        # 检查是否已完成所有模型的评测
        if bench.eval_status == "success" and bench.meta:
            existing_results = bench.meta.get("eval_results", {})
            if is_multi_model:
                # 多模型模式下检查是否所有模型都已完成
                all_done = all(m.model_name_or_path in existing_results for m in model_configs)
                if all_done:
                    self.logger.info(f"[{bench.bench_name}] 所有模型已评测完成，跳过")
                    state.eval_cursor = cursor + 1
                    return state
            else:
                # 单模型模式
                if bench.meta.get("eval_result"):
                    self.logger.info(f"[{bench.bench_name}] 已评测成功，跳过")
                    state.eval_cursor = cursor + 1
                    return state

        if not bench.dataset_cache:
            self.logger.warning(f"[{bench.bench_name}] 缺少 dataset_cache，跳过")
            bench.eval_status = "failed"
            approved = list(getattr(state, "approved_warning_ids", []) or [])
            confirm_id = "PreEvalReviewNode_confirm"
            approved = [x for x in approved if x != confirm_id]
            return Command(
                goto="PreEvalReviewNode",
                update={
                    "approved_warning_ids": approved,
                    "waiting_for_human": True,
                    "error_flag": True,
                    "error_msg": f"[{bench.bench_name}] 缺少dataset_cache，请检查下载步骤后重试评测。",
                },
            )

        if not bench.bench_dataflow_eval_type:
            self.logger.warning(f"[{bench.bench_name}] 缺少 eval_type，跳过")
            bench.eval_status = "failed"
            approved = list(getattr(state, "approved_warning_ids", []) or [])
            confirm_id = "PreEvalReviewNode_confirm"
            approved = [x for x in approved if x != confirm_id]
            return Command(
                goto="PreEvalReviewNode",
                update={
                    "approved_warning_ids": approved,
                    "waiting_for_human": True,
                    "error_flag": True,
                    "error_msg": f"[{bench.bench_name}] 缺少eval_type，请修正Key Mapping/任务类型后重试评测。",
                },
            )
        if str(bench.bench_dataflow_eval_type).strip() not in VALID_EVAL_TYPES:
            self.logger.warning(f"[{bench.bench_name}] 跳过不支持的 eval_type={bench.bench_dataflow_eval_type}")
            bench.eval_status = "failed"
            state.eval_cursor = cursor + 1
            return state

        # === 执行评测 ===
        if not bench.meta:
            bench.meta = {}

        # 初始化 eval_results 字典（多模型结果存储）
        if "eval_results" not in bench.meta:
            bench.meta["eval_results"] = {}

        bench.eval_status = "running"
        bench.meta["eval_progress"] = {
            "bench_name": bench.bench_name,
            "stage": "queued",
            "generated": 0,
            "total": 0,
            "percent": 0.0,
            "models": [m.model_name_or_path for m in model_configs],
            "completed_models": [],
        }

        # 并行执行多模型评测
        if is_multi_model:
            results = await self._run_multi_model_eval(
                bench, model_configs, tool, thread_id
            )
        else:
            # 单模型直接执行
            results = await self._run_single_model_eval(
                bench, model_configs[0], tool, thread_id
            )

        # 处理结果
        all_success = True
        for model_name, result in results.items():
            if result.get("success"):
                if is_multi_model:
                    bench.meta["eval_results"][model_name] = {
                        "stats": result["stats"],
                        "detail_path": result["detail_path"],
                    }
                else:
                    bench.meta["eval_result"] = result["stats"]
                    bench.meta["eval_detail_path"] = result["detail_path"]
            else:
                all_success = False
                self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 评测失败: {result.get('error')}")

        if all_success:
            bench.eval_status = "success"
            # 使用第一个模型的结果进行 key mapping 设置
            first_result = list(results.values())[0]
            if first_result.get("success"):
                self._set_key_mapping(bench, first_result.get("key_mapping", {}))
            self.logger.info(f"[{bench.bench_name}] 评测完成")
        else:
            bench.eval_status = "failed"
            bench.meta["eval_error"] = "部分模型评测失败"

        state.eval_cursor = cursor + 1
        if thread_id and state.eval_cursor >= len(benches):
            clear_progress(thread_id)
        return state

    async def _run_single_model_eval(
        self, bench, model_config: ModelConfig, tool, thread_id: Optional[str]
    ) -> dict:
        """执行单模型评测"""
        model_name = model_config.model_name_or_path

        def _on_progress(p: dict):
            if not bench.meta:
                bench.meta = {}
            bench.meta["eval_progress"] = p
            if thread_id:
                set_progress(thread_id, p)

        try:
            self.logger.info(f"[{bench.bench_name}] 开始评测模型: {model_name}")
            result = tool.run_eval(bench, model_config, progress_callback=_on_progress)

            if thread_id:
                set_progress(thread_id, {
                    "bench_name": bench.bench_name,
                    "stage": "done",
                    "generated": int((bench.meta.get("eval_progress") or {}).get("generated") or 0),
                    "total": int((bench.meta.get("eval_progress") or {}).get("total") or 0),
                    "percent": 100.0,
                })

            return {model_name: {
                "success": True,
                "stats": result["stats"],
                "detail_path": result["detail_path"],
                "key_mapping": result.get("key_mapping", {}),
            }}
        except Exception as e:
            self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 评测失败: {e}")
            return {model_name: {"success": False, "error": str(e)}}

    async def _run_multi_model_eval(
        self, bench, model_configs: List[ModelConfig], tool, thread_id: Optional[str]
    ) -> dict:
        """并行执行多模型评测"""
        self.logger.info(f"[{bench.bench_name}] 并行评测 {len(model_configs)} 个模型")

        async def eval_single(model_config: ModelConfig) -> tuple:
            """单个模型评测的异步任务"""
            model_name = model_config.model_name_or_path

            def _on_progress(p: dict):
                # 每个模型独立更新进度，用 {thread_id}:{model_name} 做 key
                if thread_id:
                    p_with_model = {**p, "model_name": model_name}
                    set_progress(f"{thread_id}:{model_name}", p_with_model)
                # 同时更新 bench.meta 供前端 state 同步
                if not bench.meta:
                    bench.meta = {}
                bench.meta["eval_progress"] = p

            try:
                # 在线程池中运行同步的评测方法
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: tool.run_eval(bench, model_config, progress_callback=_on_progress)
                )

                # 更新完成的模型列表
                completed = bench.meta.get("eval_progress", {}).get("completed_models", [])
                completed.append(model_name)
                bench.meta["eval_progress"]["completed_models"] = completed

                # 标记该模型完成
                if thread_id:
                    set_progress(f"{thread_id}:{model_name}", {
                        "bench_name": bench.bench_name,
                        "model_name": model_name,
                        "stage": "done",
                        "generated": int((bench.meta.get("eval_progress") or {}).get("generated") or 0),
                        "total": int((bench.meta.get("eval_progress") or {}).get("total") or 0),
                        "percent": 100.0,
                    })

                return model_name, {
                    "success": True,
                    "stats": result["stats"],
                    "detail_path": result["detail_path"],
                    "key_mapping": result.get("key_mapping", {}),
                }
            except Exception as e:
                self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 评测失败: {e}")
                return model_name, {"success": False, "error": str(e)}

        # 并行执行所有模型评测
        tasks = [eval_single(m) for m in model_configs]
        results_list = await asyncio.gather(*tasks)

        # 转换为字典
        results = dict(results_list)

        if thread_id:
            set_progress(thread_id, {
                "bench_name": bench.bench_name,
                "stage": "done",
                "generated": 0,
                "total": 0,
                "percent": 100.0,
                "models": [m.model_name_or_path for m in model_configs],
                "completed_models": list(results.keys()),
            })

        return results

    def _set_key_mapping(self, bench, final_key_mapping: dict):
        """设置 key mapping 信息"""
        eval_type = bench.bench_dataflow_eval_type

        # 确定 Pred Key
        default_pred_key = "generated_ans"
        mapped_pred_key = final_key_mapping.get("input_pred_key")
        pred_key = mapped_pred_key if mapped_pred_key else default_pred_key

        # 确定 Ref Key
        ref_key = None
        if eval_type == "key2_qa":
            ref_key = final_key_mapping.get("input_target_key")
        elif eval_type == "key2_q_ma":
            ref_key = final_key_mapping.get("input_targets_key")
        elif eval_type == "key3_q_choices_a":
            ref_key = final_key_mapping.get("input_label_key")
            pred_key = "eval_pred"
        elif eval_type == "key3_q_choices_as":
            ref_key = final_key_mapping.get("input_labels_key")
            pred_key = "eval_pred"
        elif eval_type == "key3_q_a_rejected":
            ref_key = final_key_mapping.get("input_better_key")
        elif eval_type == "key1_text_score":
            ref_key = None
            if final_key_mapping.get("input_text_key"):
                pred_key = final_key_mapping.get("input_text_key")

        if ref_key:
            bench.meta["ref_key"] = ref_key
            self.logger.info(f"[{bench.bench_name}] Set ref_key='{ref_key}'")

        bench.meta["pred_key"] = pred_key
        self.logger.info(f"[{bench.bench_name}] Set pred_key='{pred_key}'")

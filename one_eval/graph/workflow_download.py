import asyncio
from pathlib import Path
import json
import time
from dataclasses import asdict, is_dataclass

from langgraph.graph import START, END
from langgraph.types import Command

from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder

# 导入新的节点
from one_eval.nodes.dataset_structure_node import DatasetStructureNode
from one_eval.nodes.bench_config_recommend_node import BenchConfigRecommendNode
from one_eval.nodes.download_node import DownloadNode

from one_eval.utils.checkpoint import get_checkpointer
from one_eval.utils.deal_json import _save_state_json, _restore_state_from_snap
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow-Download")


def build_download_workflow(checkpointer=None):
    """
    Download Workflow (New):
    START -> DatasetStructureNode -> BenchConfigRecommendNode -> DownloadNode -> END
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="DatasetStructureNode",  # 从解析结构开始
    )

    node1 = DatasetStructureNode()
    node2 = BenchConfigRecommendNode()
    node3 = DownloadNode()

    builder.add_node(name=node1.name, func=node1.run)
    builder.add_node(name=node2.name, func=node2.run)
    builder.add_node(name=node3.name, func=node3.run)

    # 线性连接
    builder.add_edge(START, node1.name)
    builder.add_edge(node1.name, node2.name)
    builder.add_edge(node2.name, node3.name)
    builder.add_edge(node3.name, END)

    return builder.build(checkpointer=checkpointer)


async def run_download_only(thread_id: str, mode: str = "debug"):
    # === ckpt path ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    db_path = project_root / "checkpoints" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_checkpointer(db_path, mode) as checkpointer:
        graph = build_download_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 1) 检查是否存在先前的 ckpt（通常是 nl2bench 阶段生成的 benches 列表）
        snap = None
        try:
            snap = await graph.aget_state(config)
        except Exception:
            snap = None

        has_ckpt = snap is not None and (
            (getattr(snap, "next", None) not in (None, ())) or
            (getattr(snap, "values", None) not in (None, {}))
        )
        log.info(f"[download-only] thread_id={thread_id} has_ckpt={has_ckpt}")

        if not has_ckpt:
            log.error("[download-only] 未找到 ckpt：请确保已运行前置流程（如 nl2bench）生成 bench 列表。")
            # 在 debug 模式下，如果只是测试代码逻辑，你可能希望手动构造一个 state
            # 但这里我们保持严格，要求有 ckpt
            return None

        # 2) 恢复 State
        snap = await graph.aget_state(config)
        values = getattr(snap, "values", {}) or {}
        
        # 将 snapshot 转换为 NodeState 对象
        state0 = _restore_state_from_snap(values)

        # 3) 执行 Workflow
        # 注意：这里传入 state0 会作为初始输入，图会从 entry_point (DatasetStructureNode) 开始运行
        # 并基于 config 中的 thread_id 继续读写 checkpoint
        out = await graph.ainvoke(state0, config=config)

        if mode == "run":
            results_dir = project_root / "outputs"
            filename = f"download_{thread_id}_{int(time.time())}.json"
            _save_state_json(out, results_dir, filename)

        return out


if __name__ == "__main__":
    # 使用你提到的已测试过的 thread_id
    asyncio.run(run_download_only(thread_id="demo_run_006", mode="debug"))

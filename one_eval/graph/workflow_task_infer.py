import asyncio
from pathlib import Path
import json
import time

from langgraph.graph import START, END

from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder

# 导入节点
from one_eval.nodes.dataset_keys_node import DatasetKeysNode
from one_eval.nodes.bench_task_infer_node import BenchTaskInferNode

from one_eval.utils.checkpoint import get_checkpointer
from one_eval.utils.deal_json import _save_state_json, _restore_state_from_snap
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow-TaskInfer")


def build_task_infer_workflow(checkpointer=None):
    """
    Task Infer Workflow:
    START -> DatasetKeysNode -> BenchTaskInferNode -> END
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="DatasetKeysNode",
    )

    node1 = DatasetKeysNode()
    node2 = BenchTaskInferNode()

    builder.add_node(name=node1.name, func=node1.run)
    builder.add_node(name=node2.name, func=node2.run)

    builder.add_edge(START, node1.name)
    builder.add_edge(node1.name, node2.name)
    builder.add_edge(node2.name, END)

    return builder.build(checkpointer=checkpointer)


async def run_task_infer(thread_id: str, mode: str = "debug"):
    # === ckpt path ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    db_path = project_root / "checkpoints" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_checkpointer(db_path, mode) as checkpointer:
        graph = build_task_infer_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 1) 必须有 ckpt (从 Download 阶段继承)
        snap = None
        try:
            snap = await graph.aget_state(config)
        except Exception:
            snap = None

        has_ckpt = snap is not None and (
            (getattr(snap, "next", None) not in (None, ())) or
            (getattr(snap, "values", None) not in (None, {}))
        )
        log.info(f"[task-infer] thread_id={thread_id} has_ckpt={has_ckpt}")

        if not has_ckpt:
            log.error("[task-infer] 未找到 ckpt：请先运行 workflow_download.py。")
            return None

        # 2) 恢复 State
        snap = await graph.aget_state(config)
        values = getattr(snap, "values", {}) or {}
        
        state0 = _restore_state_from_snap(values)

        # 3) 执行 Workflow
        out = await graph.ainvoke(state0, config=config)

        if mode == "run":
            results_dir = project_root / "outputs"
            filename = f"task_infer_{thread_id}_{int(time.time())}.json"
            _save_state_json(out, results_dir, filename)

        return out


if __name__ == "__main__":
    # 使用与 download workflow 相同的 thread_id 以继承状态
    asyncio.run(run_task_infer(thread_id="demo_run_006", mode="run"))

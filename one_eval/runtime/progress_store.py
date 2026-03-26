from __future__ import annotations

from typing import Any, Dict, List, Optional
from threading import Lock

_LOCK = Lock()
_PROGRESS: Dict[str, Dict[str, Any]] = {}


def set_progress(thread_id: str, payload: Dict[str, Any]) -> None:
    """设置进度（支持带模型后缀的 key，如 {thread_id}:{model_name}）"""
    if not thread_id:
        return
    with _LOCK:
        _PROGRESS[thread_id] = dict(payload or {})


def get_progress(thread_id: str) -> List[Dict[str, Any]]:
    """获取 thread_id 下的所有进度（包括 {thread_id}:{model_name} 子 key）"""
    if not thread_id:
        return []
    prefix = thread_id + ":"
    with _LOCK:
        # 先收集精确匹配和前缀匹配
        results = []
        # 精确匹配
        val = _PROGRESS.get(thread_id)
        if isinstance(val, dict):
            results.append(dict(val))
        # 前缀匹配（多模型进度）
        for key, val in _PROGRESS.items():
            if key.startswith(prefix) and isinstance(val, dict):
                results.append(dict(val))
        return results


def clear_progress(thread_id: str) -> None:
    """清除 thread_id 下的所有进度"""
    if not thread_id:
        return
    prefix = thread_id + ":"
    with _LOCK:
        _PROGRESS.pop(thread_id, None)
        # 同时清除子 key
        keys_to_remove = [k for k in _PROGRESS if k.startswith(prefix)]
        for k in keys_to_remove:
            del _PROGRESS[k]

from __future__ import annotations
import asyncio
from typing import Dict, Callable, Any, List, Optional
from langchain_core.tools import Tool
from one_eval.logger import get_logger

log = get_logger(__name__)

class ToolManager:
    """支持 pre_tool + post_tool + 执行前置工具"""

    def __init__(self):
        # role -> name -> func(state) or async func(state)
        self.role_pre_tools: Dict[str, Dict[str, Callable]] = {}
        # role -> [Tool]
        self.role_post_tools: Dict[str, List[Tool]] = {}

    # ---------- pre tools ----------
    def register_pre_tool(self, *, role: str, name: str, func: Callable):
        if role not in self.role_pre_tools:
            self.role_pre_tools[role] = {}
        self.role_pre_tools[role][name] = func
        log.info(f"[ToolManager] 注册 pre_tool: role={role}, name={name}")

    async def execute_pre_tools(self, role: str, state: Any) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        tools = self.role_pre_tools.get(role, {})
        for name, func in tools.items():
            try:
                if asyncio.iscoroutinefunction(func):
                    results[name] = await func(state)
                else:
                    results[name] = func(state)
            except Exception as e:
                log.error(f"[ToolManager] pre_tool 失败: role={role}, name={name}, err={e}")
                results[name] = None
        return results

    # ---------- post tools ----------
    def register_post_tool(self, tool: Tool, role: str):
        if role not in self.role_post_tools:
            self.role_post_tools[role] = []
        self.role_post_tools[role].append(tool)
        log.info(f"[ToolManager] 注册 post_tool: role={role}, name={tool.name}")

    def get_post_tools(self, role: str) -> List[Tool]:
        return self.role_post_tools.get(role, [])

_tool_manager: Optional[ToolManager] = None

def get_tool_manager() -> ToolManager:
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager

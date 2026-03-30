# one_eval/judges/prompt_builder.py
"""组装裁判模型的完整 Prompt（三层结构：执行标准 + 原始输入 + 模型输出）。"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from one_eval.logger import get_logger

log = get_logger("PromptBuilder")


def build_judge_messages(
    scoring_prompt: str,
    question: str,
    body: str,
    think: Optional[str] = None,
    context: Optional[str] = None,
) -> List[BaseMessage]:
    """组装裁判评分的完整消息列表。

    三层结构:
    - System: 用户的执行标准 Prompt（评分维度、尺度、输出结构）
    - Human: 原始输入 + 模型输出

    Args:
        scoring_prompt: 用户的执行标准 Prompt
        question: 原始问题/Prompt
        body: 模型正文输出
        think: 模型 think 输出（如有）
        context: 额外上下文（如有）

    Returns:
        LangChain Message 列表，可直接传给 CustomLLMCaller
    """
    system_content = scoring_prompt.strip()

    # 构建 Human 消息
    parts: List[str] = []

    # 第一部分：原始输入
    parts.append("【原始输入 Prompt】")
    parts.append(question.strip())

    if context and context.strip():
        parts.append("")
        parts.append("【上下文】")
        parts.append(context.strip())

    # 第二部分：模型输出
    parts.append("")
    parts.append("【模型输出】")

    if think and think.strip():
        parts.append("")
        parts.append("— 推理过程 (think) —")
        parts.append(think.strip())

    parts.append("")
    parts.append("— 回答正文 (body) —")
    parts.append(body.strip())

    human_content = "\n".join(parts)

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

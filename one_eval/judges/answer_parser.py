# one_eval/judges/answer_parser.py
"""从模型输出 (generated_ans) 中解析 think 和 body 部分。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from one_eval.logger import get_logger

log = get_logger("AnswerParser")

# --- 常见 think 标签模式 ---
# 模型输出中 <think...> 通常没有闭合标签，正文紧跟其后
# 格式1: <think...>\n正文
# 格式2: <think...>正文</think...>（罕见但存在）
# 格式3: <answer>正文</answer>
# 格式4: 纯文本

_RE_THINK_OPEN = re.compile(
    r"<think[^>]*>", re.IGNORECASE
)
_RE_THINK_CLOSE = re.compile(
    r"</think[^>]*>", re.IGNORECASE
)
_RE_ANSWER_OPEN = re.compile(
    r"<answer[^>]*>", re.IGNORECASE
)
_RE_ANSWER_CLOSE = re.compile(
    r"</answer[^>]*>", re.IGNORECASE
)


@dataclass
class ParsedAnswer:
    """模型输出的解析结果。"""
    think: Optional[str] = None
    body: str = ""


def parse_answer(generated_ans: str) -> ParsedAnswer:
    """从 generated_ans 解析 think 和 body。

    解析优先级:
    1. <think...> + </think...> 闭合标签 → 提取中间为 think，剩余为 body
    2. <think...> 无闭合 → 提取到标签结束位置之后的所有内容，尝试按空行/换行切分
    3. <answer>...</answer> → 提取为 body
    4. 纯文本 → 整体作为 body
    """
    if not generated_ans:
        return ParsedAnswer(body="")

    text = generated_ans.strip()

    # 检测 <think...> 标签
    think_match = _RE_THINK_OPEN.search(text)
    if think_match:
        think_start = think_match.end()
        think_close_match = _RE_THINK_CLOSE.search(text, think_start)

        if think_close_match:
            # 有闭合标签
            think_content = text[think_start:think_close_match.start()].strip()
            # body = 闭合标签之后的内容
            remaining = text[think_close_match.end():].strip()
        else:
            # 无闭合标签 — think 内容从标签后到正文开始的分界
            # 常见模式：think内容后跟空行、或直接跟正文
            after_think = text[think_start:]
            # 尝试按连续空行（\n\n）分割
            parts = re.split(r"\n{2,}", after_think, maxsplit=1)
            if len(parts) >= 2:
                think_content = parts[0].strip()
                remaining = parts[1].strip()
            else:
                # 没有空行分割，尝试按单个换行分割，取第一段为 think
                # 但如果 think 很短（<50字），可能整段都是 body
                lines = after_think.split("\n", 1)
                if len(lines) >= 2 and len(lines[0].strip()) > 20:
                    think_content = lines[0].strip()
                    remaining = lines[1].strip()
                else:
                    # 整段当作 think，body 为空
                    think_content = after_think.strip()
                    remaining = ""

        # 检查 body 中是否包含 <answer>...</answer>
        if remaining:
            answer_match = _RE_ANSWER_OPEN.search(remaining)
            if answer_match:
                answer_close = _RE_ANSWER_CLOSE.search(remaining, answer_match.end())
                if answer_close:
                    body = remaining[answer_match.end():answer_close.start()].strip()
                else:
                    body = remaining[answer_match.end():].strip()
            else:
                body = remaining
        else:
            body = ""

        return ParsedAnswer(think=think_content or None, body=body or "")

    # 没有 <think...> 标签，检查 <answer>...</answer>
    answer_match = _RE_ANSWER_OPEN.search(text)
    if answer_match:
        answer_close = _RE_ANSWER_CLOSE.search(text, answer_match.end())
        if answer_close:
            body = text[answer_match.end():answer_close.start()].strip()
        else:
            body = text[answer_match.end():].strip()
        # answer 之前的内容可以作为 think（如果有意义）
        before_answer = text[:answer_match.start()].strip()
        think = before_answer if len(before_answer) > 20 else None
        return ParsedAnswer(think=think, body=body or "")

    # 纯文本 — 整体作为 body
    return ParsedAnswer(body=text)

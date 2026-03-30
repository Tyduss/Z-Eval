# one_eval/judges/score_parser.py
"""解析裁判模型的返回结果，提取结构化评分。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from one_eval.logger import get_logger

log = get_logger("ScoreParser")

# 匹配 ```json ... ``` 或 ``` ... ``` 代码块
_RE_JSON_BLOCK = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
# 匹配 { ... } JSON 对象（可能跨行）
_RE_JSON_OBJECT = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


@dataclass
class ParsedScore:
    """裁判输出的结构化评分。字段由用户的执行标准 Prompt 动态定义。"""
    # 典型分数字段（动态解析，以下为常见示例）
    think_score: Optional[float] = None
    body_score: Optional[float] = None
    overall_score: Optional[float] = None
    # 问题诊断字段
    critical_issue: Optional[str] = None
    other_issues: Optional[List[str]] = None
    remark: Optional[str] = None
    # 所有动态解析出的字段
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    # 元信息
    raw_output: str = ""
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，用于 JSONL 存储。"""
        d: Dict[str, Any] = {}
        if self.think_score is not None:
            d["think_score"] = self.think_score
        if self.body_score is not None:
            d["body_score"] = self.body_score
        if self.overall_score is not None:
            d["overall_score"] = self.overall_score
        if self.critical_issue is not None:
            d["critical_issue"] = self.critical_issue
        if self.other_issues is not None:
            d["other_issues"] = self.other_issues
        if self.remark is not None:
            d["remark"] = self.remark
        d.update(self.extra_fields)
        if self.parse_error:
            d["parse_error"] = self.parse_error
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_output: str = "") -> "ParsedScore":
        """从字典反序列化。"""
        known_fields = {"think_score", "body_score", "overall_score",
                        "critical_issue", "other_issues", "remark"}
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            think_score=_to_float(data.get("think_score")),
            body_score=_to_float(data.get("body_score")),
            overall_score=_to_float(data.get("overall_score")),
            critical_issue=data.get("critical_issue"),
            other_issues=data.get("other_issues"),
            remark=data.get("remark"),
            extra_fields=extra,
            raw_output=raw_output,
        )


def _to_float(value: Any) -> Optional[float]:
    """安全转换为 float。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_judge_output(raw_output: str) -> ParsedScore:
    """解析裁判模型的原始输出，提取结构化评分。

    解析策略（优先级递减）:
    1. 提取 ```json ... ``` 代码块 → JSON 解析
    2. 提取 { ... } JSON 对象 → JSON 解析
    3. JSON 解析失败 → 尝试正则提取 _score 后缀的数字字段
    4. 全部失败 → 标记 parse_error，保留 raw_output

    Args:
        raw_output: 裁判模型的原始文本输出

    Returns:
        ParsedScore 结构化评分结果
    """
    if not raw_output or not raw_output.strip():
        return ParsedScore(raw_output=raw_output, parse_error="empty_output")

    raw_output = raw_output.strip()

    # 策略1: 提取 JSON 代码块
    json_str = _extract_json_block(raw_output)

    if not json_str:
        # 策略2: 提取 JSON 对象
        json_str = _extract_json_object(raw_output)

    if json_str:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return ParsedScore.from_dict(data, raw_output=raw_output)
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse failed: {e}")

    # 策略3: 正则提取数字分数
    score_fields = _extract_score_fields(raw_output)
    if score_fields:
        result = ParsedScore(raw_output=raw_output, parse_error="json_parse_failed_fallback_regex")
        for key, val in score_fields.items():
            if key == "think_score":
                result.think_score = val
            elif key == "body_score":
                result.body_score = val
            elif key == "overall_score":
                result.overall_score = val
            else:
                result.extra_fields[key] = val
        return result

    # 策略4: 全部失败
    return ParsedScore(raw_output=raw_output, parse_error="no_parseable_content")


def _extract_json_block(text: str) -> Optional[str]:
    """提取 ```json ... ``` 或 ``` ... ``` 代码块。"""
    match = _RE_JSON_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return None


def _extract_json_object(text: str) -> Optional[str]:
    """提取文本中第一个 { ... } JSON 对象。"""
    match = _RE_JSON_OBJECT.search(text)
    if match:
        return match.group(0)
    return None


def _extract_score_fields(text: str) -> Dict[str, float]:
    """正则提取形如 key_score: 4.5 或 "key_score": 4 的数字。"""
    scores: Dict[str, float] = {}
    # 匹配 "xxx_score": 数字 或 xxx_score: 数字 或 xxx_score：数字
    pattern = re.compile(
        r'["\']?(\w+_score)["\']?\s*[:：]\s*(\d+(?:\.\d+)?)',
        re.IGNORECASE
    )
    for match in pattern.finditer(text):
        key = match.group(1)
        val = float(match.group(2))
        scores[key] = val
    return scores

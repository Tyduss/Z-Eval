# one_eval/judges/score_aggregator.py
"""评分聚合统计 — 单模型统计 + 多模型对比排名 + 高频问题分析。"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from one_eval.judges.llm_judge import JudgeResult
from one_eval.logger import get_logger

log = get_logger("ScoreAggregator")

# 常见分数字段名（用于自动检测）
_KNOWN_SCORE_FIELDS = [
    "overall_score", "body_score", "think_score",
    "accuracy_score", "relevance_score", "coherence_score",
    "fluency_score", "completeness_score",
]


def _detect_score_fields(results: List[JudgeResult]) -> List[str]:
    """自动检测评分结果中的分数字段。"""
    fields: List[str] = []
    for r in results:
        if r.error or not r.score:
            continue
        d = r.score.to_dict()
        for k, v in d.items():
            if k in ("parse_error", "raw_output", "extra_fields"):
                continue
            if isinstance(v, (int, float)) and k not in fields:
                fields.append(k)
    # 按已知字段优先级排序
    ordered = [f for f in _KNOWN_SCORE_FIELDS if f in fields]
    ordered += [f for f in fields if f not in ordered]
    return ordered


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.mean(values), 4)


def _safe_median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(statistics.median(values), 4)


def _safe_stdev(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return round(statistics.stdev(values), 4)


@dataclass
class ModelJudgeStats:
    """单个模型的裁判评分统计。"""
    model_name: str
    total_samples: int = 0
    success_count: int = 0
    error_count: int = 0
    # 分数统计（动态字段）
    avg_scores: Dict[str, float] = field(default_factory=dict)
    median_scores: Dict[str, float] = field(default_factory=dict)
    stdev_scores: Dict[str, float] = field(default_factory=dict)
    # 分数分布: {field_name: {score_value: count}}
    score_distributions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # 问题统计
    top_critical_issues: List[str] = field(default_factory=list)
    top_other_issues: List[str] = field(default_factory=list)
    # 排名（多模型对比时由 aggregator 填充）
    ranking: int = 0


@dataclass
class ComparisonStats:
    """多模型对比统计。"""
    model_rankings: List[Dict[str, Any]] = field(default_factory=list)
    # 跨模型高频问题（所有模型合并统计）
    global_top_critical_issues: List[Dict[str, Any]] = field(default_factory=list)
    global_top_other_issues: List[Dict[str, Any]] = field(default_factory=list)


class ScoreAggregator:

    def __init__(self, top_issues_n: int = 10):
        self.top_issues_n = top_issues_n

    def aggregate_model(self, results: List[JudgeResult]) -> ModelJudgeStats:
        """对单个模型的评分结果进行聚合统计。"""
        if not results:
            return ModelJudgeStats(model_name="unknown")

        model_name = results[0].model_name
        total = len(results)
        success = [r for r in results if not r.error]
        errors = [r for r in results if r.error]

        stats = ModelJudgeStats(
            model_name=model_name,
            total_samples=total,
            success_count=len(success),
            error_count=len(errors),
        )

        # 自动检测分数字段
        score_fields = _detect_score_fields(success)
        if not score_fields:
            log.warning(f"[{model_name}] No score fields detected")
            return stats

        # 逐字段统计
        for sf in score_fields:
            values: List[float] = []
            dist: Counter = Counter()
            for r in success:
                if r.score is None:
                    continue
                d = r.score.to_dict()
                v = d.get(sf)
                if isinstance(v, (int, float)):
                    values.append(float(v))
                    dist[str(v)] += 1

            stats.avg_scores[sf] = _safe_mean(values) or 0.0
            stats.median_scores[sf] = _safe_median(values) or 0.0
            stats.stdev_scores[sf] = _safe_stdev(values) or 0.0
            stats.score_distributions[sf] = dict(dist)

        # 问题统计
        critical_issues: List[str] = []
        other_issues: List[str] = []
        for r in success:
            if r.score is None:
                continue
            if r.score.critical_issue and r.score.critical_issue.lower() not in ("无", "none", "n/a", ""):
                critical_issues.append(r.score.critical_issue)
            if r.score.other_issues and isinstance(r.score.other_issues, list):
                other_issues.extend(r.score.other_issues)

        stats.top_critical_issues = [
            item for item, _ in Counter(critical_issues).most_common(self.top_issues_n)
        ]
        stats.top_other_issues = [
            item for item, _ in Counter(other_issues).most_common(self.top_issues_n)
        ]

        return stats

    def aggregate_comparison(
        self, all_results: Dict[str, List[JudgeResult]]
    ) -> ComparisonStats:
        """多模型横向对比排名 + 跨模型问题共性分析。"""
        # 各模型统计
        model_stats: Dict[str, ModelJudgeStats] = {}
        for model_name, results in all_results.items():
            model_stats[model_name] = self.aggregate_model(results)

        # 确定主排序字段（优先 overall_score，其次第一个分数字段）
        primary_field = "overall_score"
        all_score_fields = set()
        for ms in model_stats.values():
            all_score_fields.update(ms.avg_scores.keys())
        if primary_field not in all_score_fields and all_score_fields:
            primary_field = sorted(
                all_score_fields,
                key=lambda f: _KNOWN_SCORE_FIELDS.index(f) if f in _KNOWN_SCORE_FIELDS else 999
            )[0]

        # 按主字段降序排名
        ranked = sorted(
            model_stats.values(),
            key=lambda ms: ms.avg_scores.get(primary_field, 0.0),
            reverse=True,
        )
        for i, ms in enumerate(ranked):
            ms.ranking = i + 1

        # 构建排名表
        model_rankings = []
        for ms in ranked:
            model_rankings.append({
                "ranking": ms.ranking,
                "model_name": ms.model_name,
                "total_samples": ms.total_samples,
                "success_count": ms.success_count,
                "error_count": ms.error_count,
                "avg_scores": ms.avg_scores,
                "median_scores": ms.median_scores,
            })

        # 跨模型高频问题（合并所有模型的问题）
        all_critical: List[str] = []
        all_other: List[str] = []
        for ms in model_stats.values():
            all_critical.extend(ms.top_critical_issues)
            all_other.extend(ms.top_other_issues)

        global_top_critical = [
            {"issue": item, "count": count}
            for item, count in Counter(all_critical).most_common(self.top_issues_n)
        ]
        global_top_other = [
            {"issue": item, "count": count}
            for item, count in Counter(all_other).most_common(self.top_issues_n)
        ]

        return ComparisonStats(
            model_rankings=model_rankings,
            global_top_critical_issues=global_top_critical,
            global_top_other_issues=global_top_other,
        )

    def save_summary(
        self,
        model_stats: Dict[str, ModelJudgeStats],
        comparison: ComparisonStats,
        output_dir: Path,
    ) -> str:
        """保存统计摘要到 JSON 文件。返回文件路径。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "model_stats": {
                name: asdict(ms) for name, ms in model_stats.items()
            },
            "comparison": asdict(comparison),
        }

        path = output_dir / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

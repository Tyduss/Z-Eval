# hf_bench_probe_tool.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SplitInfo:
    name: str
    num_examples: Optional[int] = None


@dataclass
class SubsetInfo:
    subset: str
    splits: List[SplitInfo]
    features: Optional[List[str]] = None


@dataclass
class RepoStructure:
    repo_id: str
    revision: Optional[str]
    subsets: List[SubsetInfo]
    ok: bool
    error: Optional[str] = None


class HFDatasetStructureTool:
    """
    Tool: 探测 HF dataset repo 的所有 subset 和 split

    设计目标
    - 不做任何 test/validation/train 的硬编码选择
    - 返回完整候选空间给 agent 决策
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.revision = revision
        self.trust_remote_code = trust_remote_code

    def _import_datasets(self):
        try:
            from datasets import (
                get_dataset_config_names,
                get_dataset_split_names,
                load_dataset_builder,
            )
            return get_dataset_config_names, get_dataset_split_names, load_dataset_builder
        except Exception as e:
            raise RuntimeError(
                "缺少 datasets 依赖或导入失败 请先 pip install datasets"
            ) from e

    def probe(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        include_features: bool = True,
        include_num_examples: bool = True,
    ) -> Dict[str, Any]:
        """
        返回结构
        {
          repo_id,
          revision,
          ok,
          error,
          subsets: [
            { subset, features, splits: [{name, num_examples}] }
          ]
        }
        """
        revision = revision or self.revision
        try:
            get_cfgs, get_splits, load_builder = self._import_datasets()
        except Exception as e:
            res = RepoStructure(
                repo_id=repo_id,
                revision=revision,
                subsets=[],
                ok=False,
                error=str(e),
            )
            return asdict(res)

        try:
            cfgs: List[str] = []
            try:
                cfgs = list(get_cfgs(repo_id, revision=revision, token=self.hf_token))
            except TypeError:
                cfgs = list(get_cfgs(repo_id))
            except Exception:
                cfgs = []

            # 有些数据集没有显式 config 也能 load, 这里统一用 default 占位
            if not cfgs:
                cfgs = ["default"]

            subsets: List[SubsetInfo] = []

            for cfg in cfgs:
                # splits
                split_names: List[str] = []
                try:
                    try:
                        split_names = list(get_splits(repo_id, config_name=cfg, revision=revision, token=self.hf_token))
                    except TypeError:
                        split_names = list(get_splits(repo_id, config_name=cfg))
                except Exception:
                    split_names = []

                # features + num_examples 可选从 builder.info 里拿
                feats: Optional[List[str]] = None
                split_examples: Dict[str, Optional[int]] = {}

                if include_features or include_num_examples:
                    try:
                        try:
                            b = load_builder(repo_id, cfg, revision=revision, token=self.hf_token, trust_remote_code=self.trust_remote_code)
                        except TypeError:
                            b = load_builder(repo_id, cfg)
                        info = getattr(b, "info", None)

                        if include_features and info is not None and getattr(info, "features", None) is not None:
                            feats = list(info.features.keys())

                        if include_num_examples and info is not None and getattr(info, "splits", None) is not None:
                            # info.splits: dict-like, value has num_examples
                            for sname, s in info.splits.items():
                                ne = getattr(s, "num_examples", None)
                                split_examples[str(sname)] = int(ne) if isinstance(ne, int) else None
                    except Exception:
                        pass

                splits: List[SplitInfo] = []
                for s in split_names:
                    ne = split_examples.get(s) if include_num_examples else None
                    splits.append(SplitInfo(name=s, num_examples=ne))

                subsets.append(
                    SubsetInfo(
                        subset=cfg,
                        splits=splits,
                        features=feats if include_features else None,
                    )
                )

            res = RepoStructure(
                repo_id=repo_id,
                revision=revision,
                subsets=subsets,
                ok=True,
                error=None,
            )
            return asdict(res)

        except Exception as e:
            res = RepoStructure(
                repo_id=repo_id,
                revision=revision,
                subsets=[],
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )
            return asdict(res)

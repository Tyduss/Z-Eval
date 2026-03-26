from __future__ import annotations
import json
import os
from typing import List, Dict, Optional, Any
from one_eval.logger import get_logger

log = get_logger("BenchRegistry")


class BenchRegistry:
    """
    本地 Benchmark 注册中心：
    - 从 bench_gallery_public.json 加载公共数据（可同步到仓库）
    - 从 bench_gallery_local.json 加载本地映射（包含敏感路径，不入库）
    - 支持用户指定 benchmark (specific_benches)
    - 支持根据 domain 与 bench.meta.tags 匹配的自动推荐
    """

    def __init__(self, config_path: str, local_mapping_path: Optional[str] = None):
        """
        加载 bench gallery 文件

        支持两种模式：
        1. 旧模式：只传入 config_path，加载单一的 bench_gallery.json
        2. 新模式：同时传入 config_path 和 local_mapping_path
           - config_path: bench_gallery_public.json（公共数据）
           - local_mapping_path: bench_gallery_local.json（本地映射）

        bench_gallery_public.json 格式:
        {
            "benches": [
                {
                    "bench_name": "写诗测试",
                    "bench_source_url": "local://uploaded/写诗测试",  // 脱敏后的标识符
                    "meta": {
                        "is_local_upload": true,
                        ...
                    }
                },
                ...
            ]
        }

        bench_gallery_local.json 格式:
        {
            "local_mappings": {
                "写诗测试": {
                    "actual_source_url": "local://uploads/写诗测试.jsonl",
                    "dataset_cache": "/path/to/uploads/写诗测试.jsonl",
                    ...
                }
            }
        }
        """
        self.benches: List[Dict[str, Any]] = []
        self.data: Dict[str, Dict[str, Any]] = {}  # bench_name -> bench_info 的映射
        self.lower_map: Dict[str, str] = {}  # lowercase bench_name -> original bench_name
        self.local_mappings: Dict[str, Dict[str, Any]] = {}  # 本地映射数据

        # 判断是否使用双文件模式
        if local_mapping_path and os.path.exists(local_mapping_path):
            self._load_dual_file_mode(config_path, local_mapping_path)
        else:
            # 兼容旧模式：单一文件加载
            self._load_single_file_mode(config_path)

    def _load_single_file_mode(self, config_path: str):
        """旧模式：从单一文件加载"""
        if not os.path.exists(config_path):
            log.error(f"Bench gallery file not found: {config_path}")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self._parse_raw_data(raw_data, config_path)

    def _load_dual_file_mode(self, public_path: str, local_path: str):
        """新模式：从公共文件和本地映射文件加载"""
        # 1. 加载公共文件
        if not os.path.exists(public_path):
            log.error(f"Public gallery file not found: {public_path}")
            return

        with open(public_path, "r", encoding="utf-8") as f:
            public_data = json.load(f)

        # 2. 加载本地映射
        with open(local_path, "r", encoding="utf-8") as f:
            local_data = json.load(f)
            self.local_mappings = local_data.get("local_mappings", {})

        # 3. 合并数据
        self._parse_raw_data_with_local_mappings(public_data, self.local_mappings, public_path)

    def _parse_raw_data(self, raw_data: Dict, config_path: str):
        """解析原始数据（兼容旧格式）"""
        # 支持两种格式：
        # 1. {"benches": [...]} - bench_gallery.json 格式
        # 2. {"bench_name": {...}, ...} - 旧的 bench_config.json 格式（兼容）
        if isinstance(raw_data, dict) and "benches" in raw_data:
            # bench_gallery.json 格式
            self.benches = raw_data.get("benches", [])
            # 构建 bench_name -> info 的映射
            for bench in self.benches:
                bench_name = bench.get("bench_name")
                if bench_name:
                    self.data[bench_name] = bench
                    self.lower_map[bench_name.lower()] = bench_name
        elif isinstance(raw_data, dict):
            # 旧的 bench_config.json 格式（兼容）
            self.data = raw_data
            self.lower_map = {name.lower(): name for name in self.data.keys()}
            # 转换为 benches 列表格式
            for name, info in self.data.items():
                self.benches.append({"bench_name": name, **info})

        log.info(f"[BenchRegistry] Loaded {len(self.benches)} benches from {config_path}")

    def _parse_raw_data_with_local_mappings(self, public_data: Dict, local_mappings: Dict, config_path: str):
        """解析公共数据并应用本地映射"""
        if isinstance(public_data, dict) and "benches" in public_data:
            benches = public_data.get("benches", [])
            for bench in benches:
                bench_name = bench.get("bench_name")
                if not bench_name:
                    continue

                # 如果是本地上传的评测集，应用本地映射
                meta = bench.get("meta", {})
                if meta.get("is_local_upload") or bench.get("bench_source_url", "").startswith("local://uploaded"):
                    mapping = local_mappings.get(bench_name, {})
                    if mapping:
                        # 应用本地映射：恢复实际的 source_url 和 dataset_cache
                        if mapping.get("actual_source_url"):
                            bench["bench_source_url"] = mapping["actual_source_url"]
                        if mapping.get("dataset_cache"):
                            bench["dataset_cache"] = mapping["dataset_cache"]

                self.benches.append(bench)
                self.data[bench_name] = bench
                self.lower_map[bench_name.lower()] = bench_name

        log.info(f"[BenchRegistry] Loaded {len(self.benches)} benches (with {len(local_mappings)} local mappings) from {config_path}")

    # ---------------------------------------------------------
    # 辅助：名称 / alias 匹配
    # ---------------------------------------------------------
    def _match_bench_by_name_or_alias(self, query: str) -> Optional[str]:
        """支持大小写不敏感匹配 + alias 匹配"""
        if not isinstance(query, str):
            return None

        query = query.strip().lower()
        if not query:
            return None

        # bench_name 匹配
        if query in self.lower_map:
            return self.lower_map[query]

        # alias 匹配
        for bench in self.benches:
            bench_name = bench.get("bench_name")
            meta = bench.get("meta", {})
            aliases = meta.get("aliases", []) or []
            for alias in aliases:
                if not isinstance(alias, str):
                    continue
                if query == alias.lower() or query in alias.lower():
                    return bench_name

        return None

    # ---------------------------------------------------------
    # API: 支持 domain=None 的 search
    # ---------------------------------------------------------
    def search(
        self,
        specific_benches: Optional[List[str]] = None,
        domain: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        先匹配用户指定的 benchmark，再进行 domain 匹配推荐。

        specific_benches:
            - 用户点名的 benchmark 名称（不区分大小写），会和 bench_name / aliases 比对
        domain:
            - 任务领域标签，如 ["text", "math", "reasoning"]
            - 为 None 或空时，不做领域过滤（只按名称/别名匹配）
        """
        results: List[Dict] = []

        specific_benches = specific_benches or []

        # 规范化 domain；None 表示「不做领域过滤」
        if domain:
            domain_set = {d.strip().lower() for d in domain if isinstance(d, str) and d.strip()}
        else:
            domain_set = None  # 不按领域过滤

        # 如果既没有指定 bench，也没有 domain，就直接返回空，避免把全表都打出来
        if not specific_benches and domain_set is None:
            return results

        # --------------------------
        # Step 1. 处理用户指定 benchmark
        # --------------------------
        for name in specific_benches:
            matched = self._match_bench_by_name_or_alias(name)
            if matched and matched in self.data:
                bench = {
                    "bench_name": matched,
                    "source": "specified",
                    **self.data[matched],
                }
                results.append(bench)
            else:
                log.warning(f"[BenchRegistry] User specified bench '{name}' not found.")

        # --------------------------
        # Step 2. 根据 domain 匹配推荐 benchmark（使用 meta.tags）
        # --------------------------
        for bench in self.benches:
            bench_name = bench.get("bench_name")
            if not bench_name:
                continue

            # 已经被用户指定过的避免重复
            if any(r["bench_name"] == bench_name for r in results):
                continue

            # 如果没有 domain_set（即 domain=None），则不做领域过滤，直接跳过推荐逻辑
            if domain_set is None:
                continue

            # 从 meta.tags 获取标签
            meta = bench.get("meta", {})
            tags = meta.get("tags", [])
            if not tags:
                continue

            if isinstance(tags, str):
                tags_list = [tags]
            else:
                tags_list = tags

            tags_set = {t.strip().lower() for t in tags_list if isinstance(t, str) and t.strip()}

            if domain_set & tags_set:  # 有交集
                result_bench = {
                    "bench_name": bench_name,
                    "source": "local_recommend",
                    **bench,
                }
                results.append(result_bench)

        return results

    def get_all_benches(self) -> List[Dict]:
        """
        返回所有注册的 benchmark 列表 (用于 Gallery 展示)

        返回完整的 bench_gallery.json 数据结构
        """
        return self.benches

    def get_bench_by_name(self, bench_name: str) -> Optional[Dict]:
        """根据名称获取单个 benchmark 的完整信息"""
        matched = self._match_bench_by_name_or_alias(bench_name)
        if matched and matched in self.data:
            return self.data[matched]
        return None

    def add_bench(self, bench_data: Dict[str, Any], config_path: str) -> bool:
        """
        添加新的 benchmark 到注册表和文件

        Args:
            bench_data: 新 benchmark 的数据
            config_path: bench_gallery.json 文件路径

        Returns:
            成功返回 True，失败返回 False
        """
        bench_name = bench_data.get("bench_name")
        if not bench_name:
            log.error("[BenchRegistry] bench_name is required")
            return False

        # 检查是否已存在
        if bench_name.lower() in self.lower_map:
            log.warning(f"[BenchRegistry] Bench '{bench_name}' already exists")
            return False

        # 添加到内存
        self.benches.append(bench_data)
        self.data[bench_name] = bench_data
        self.lower_map[bench_name.lower()] = bench_name

        # 写入文件
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)

            if isinstance(file_data, dict) and "benches" in file_data:
                file_data["benches"].append(bench_data)
            else:
                file_data = {"benches": self.benches}

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False, indent=2)

            log.info(f"[BenchRegistry] Added bench '{bench_name}' successfully")
            return True
        except Exception as e:
            log.error(f"[BenchRegistry] Failed to save bench: {e}")
            # 回滚内存修改
            self.benches.pop()
            del self.data[bench_name]
            del self.lower_map[bench_name.lower()]
            return False

    def delete_bench(self, bench_name: str, config_path: str, local_mapping_path: Optional[str] = None) -> bool:
        """
        从注册表和文件中删除 benchmark

        Args:
            bench_name: 要删除的 benchmark 名称
            config_path: bench_gallery_public.json 文件路径
            local_mapping_path: bench_gallery_local.json 文件路径（可选）

        Returns:
            成功返回 True，失败返回 False
        """
        # 检查是否存在
        matched = self._match_bench_by_name_or_alias(bench_name)
        if not matched:
            log.warning(f"[BenchRegistry] Bench '{bench_name}' not found")
            return False

        original_name = matched

        # 从内存中删除
        self.benches = [b for b in self.benches if b.get("bench_name") != original_name]
        self.data.pop(original_name, None)
        self.lower_map.pop(original_name.lower(), None)

        # 写入公共文件
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)

            if isinstance(file_data, dict) and "benches" in file_data:
                file_data["benches"] = [b for b in file_data["benches"] if b.get("bench_name") != original_name]

                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(file_data, f, ensure_ascii=False, indent=2)

            log.info(f"[BenchRegistry] Deleted bench '{original_name}' from public gallery")
        except Exception as e:
            log.error(f"[BenchRegistry] Failed to delete bench from public file: {e}")
            return False

        # 如果提供了本地映射文件，也从中删除
        if local_mapping_path and os.path.exists(local_mapping_path):
            try:
                with open(local_mapping_path, "r", encoding="utf-8") as f:
                    local_data = json.load(f)

                if "local_mappings" in local_data and original_name in local_data["local_mappings"]:
                    del local_data["local_mappings"][original_name]

                    with open(local_mapping_path, "w", encoding="utf-8") as f:
                        json.dump(local_data, f, ensure_ascii=False, indent=2)

                    log.info(f"[BenchRegistry] Deleted bench '{original_name}' from local mappings")
            except Exception as e:
                log.error(f"[BenchRegistry] Failed to delete bench from local mappings: {e}")

        return True

    def add_local_upload_bench(
        self,
        bench_data: Dict[str, Any],
        public_path: str,
        local_mapping_path: str,
        local_mapping: Dict[str, Any]
    ) -> bool:
        """
        添加本地上传的 benchmark 到注册表和两个文件

        Args:
            bench_data: 脱敏后的 bench 数据（用于公共文件）
            public_path: bench_gallery_public.json 文件路径
            local_mapping_path: bench_gallery_local.json 文件路径
            local_mapping: 本地映射数据（包含实际路径等敏感信息）

        Returns:
            成功返回 True，失败返回 False
        """
        bench_name = bench_data.get("bench_name")
        if not bench_name:
            log.error("[BenchRegistry] bench_name is required")
            return False

        # 检查是否已存在
        if bench_name.lower() in self.lower_map:
            log.warning(f"[BenchRegistry] Bench '{bench_name}' already exists")
            return False

        # 保存到公共文件
        try:
            with open(public_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)

            if isinstance(file_data, dict) and "benches" in file_data:
                file_data["benches"].append(bench_data)
            else:
                file_data = {"benches": [bench_data]}

            with open(public_path, "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"[BenchRegistry] Failed to save to public file: {e}")
            return False

        # 保存到本地映射文件
        try:
            local_data = {"local_mappings": {}}
            if os.path.exists(local_mapping_path):
                with open(local_mapping_path, "r", encoding="utf-8") as f:
                    local_data = json.load(f)
                    if "local_mappings" not in local_data:
                        local_data["local_mappings"] = {}

            local_data["local_mappings"][bench_name] = local_mapping

            with open(local_mapping_path, "w", encoding="utf-8") as f:
                json.dump(local_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"[BenchRegistry] Failed to save to local mapping file: {e}")
            # 回滚公共文件
            try:
                with open(public_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                if isinstance(file_data, dict) and "benches" in file_data:
                    file_data["benches"] = [b for b in file_data["benches"] if b.get("bench_name") != bench_name]
                    with open(public_path, "w", encoding="utf-8") as f:
                        json.dump(file_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return False

        # 添加到内存（合并后的完整数据）
        full_bench_data = dict(bench_data)
        if local_mapping.get("actual_source_url"):
            full_bench_data["bench_source_url"] = local_mapping["actual_source_url"]
        if local_mapping.get("dataset_cache"):
            full_bench_data["dataset_cache"] = local_mapping["dataset_cache"]

        self.benches.append(full_bench_data)
        self.data[bench_name] = full_bench_data
        self.lower_map[bench_name.lower()] = bench_name
        self.local_mappings[bench_name] = local_mapping

        log.info(f"[BenchRegistry] Added local upload bench '{bench_name}' successfully")
        return True

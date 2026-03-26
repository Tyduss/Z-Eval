#!/usr/bin/env python3
"""
迁移脚本：将 bench_gallery.json 拆分为公共文件和本地映射文件

使用方法:
    python scripts/migrate_bench_gallery.py

功能:
    1. 读取现有的 bench_gallery.json
    2. 识别本地上传的评测集（通过 bench_source_url 包含 "local://uploads" 识别）
    3. 提取本地路径信息到 bench_gallery_local.json
    4. 创建脱敏的 bench_gallery_public.json
    5. 备份原始文件为 bench_gallery.json.bak
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List


def is_local_upload_bench(bench: Dict[str, Any]) -> bool:
    """判断是否为本地上传的评测集"""
    source_url = bench.get("bench_source_url", "")
    meta = bench.get("meta", {})

    # 通过 source_url 或 meta 标志判断
    return (
        source_url.startswith("local://uploads") or
        meta.get("source") == "local_upload" or
        meta.get("is_local_upload") is True
    )


def sanitize_bench_data(bench: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    脱敏处理：将 bench 数据分离为公共数据和本地映射

    Returns:
        (public_bench, local_mapping)
    """
    bench_name = bench.get("bench_name", "")
    source_url = bench.get("bench_source_url", "")
    dataset_cache = bench.get("dataset_cache", "")
    meta = bench.get("meta", {})

    # 创建公共数据（脱敏）
    public_bench = dict(bench)

    # 如果是本地上传，进行脱敏处理
    if is_local_upload_bench(bench):
        # 1. 脱敏 bench_source_url
        # 原始: "local://uploads/写诗测试.jsonl"
        # 脱敏: "local://uploaded/写诗测试"
        if source_url.startswith("local://uploads/"):
            filename = source_url.replace("local://uploads/", "")
            # 移除文件扩展名
            bench_id = os.path.splitext(filename)[0]
            public_bench["bench_source_url"] = f"local://uploaded/{bench_id}"

        # 2. 移除 dataset_cache（敏感路径）
        if dataset_cache:
            del public_bench["dataset_cache"]

        # 3. 添加标志位
        if "meta" not in public_bench:
            public_bench["meta"] = {}
        public_bench["meta"]["is_local_upload"] = True

        # 创建本地映射
        local_mapping = {
            "actual_source_url": source_url,
            "dataset_cache": dataset_cache,
            "uploaded_filename": os.path.basename(source_url.replace("local://uploads/", "")) if source_url else "",
            "migrated_at": datetime.now(timezone.utc).isoformat()
        }

        return public_bench, local_mapping
    else:
        # 非本地上传，不需要本地映射
        return public_bench, {}


def migrate_bench_gallery(
    input_path: str,
    public_output_path: str,
    local_output_path: str,
    backup: bool = True
) -> Dict[str, Any]:
    """
    执行迁移

    Args:
        input_path: 原始 bench_gallery.json 路径
        public_output_path: 公共文件输出路径
        local_output_path: 本地映射文件输出路径
        backup: 是否备份原始文件

    Returns:
        迁移统计信息
    """
    print(f"[Migration] Starting migration...")
    print(f"  Input: {input_path}")
    print(f"  Public output: {public_output_path}")
    print(f"  Local output: {local_output_path}")

    # 读取原始文件
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    benches = raw_data.get("benches", [])
    print(f"[Migration] Found {len(benches)} benches")

    # 分离数据
    public_benches: List[Dict[str, Any]] = []
    local_mappings: Dict[str, Dict[str, Any]] = {}
    local_count = 0

    for bench in benches:
        public_bench, local_mapping = sanitize_bench_data(bench)
        public_benches.append(public_bench)

        if local_mapping:
            bench_name = bench.get("bench_name", "")
            local_mappings[bench_name] = local_mapping
            local_count += 1
            print(f"  [Local] {bench_name}: {local_mapping.get('dataset_cache', 'N/A')}")

    # 备份原始文件
    if backup:
        backup_path = f"{input_path}.bak"
        if os.path.exists(backup_path):
            # 如果备份文件已存在，添加时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{input_path}.bak.{timestamp}"
        shutil.copy2(input_path, backup_path)
        print(f"[Migration] Backup created: {backup_path}")

    # 写入公共文件
    public_data = {"benches": public_benches}
    os.makedirs(os.path.dirname(public_output_path), exist_ok=True)
    with open(public_output_path, "w", encoding="utf-8") as f:
        json.dump(public_data, f, ensure_ascii=False, indent=2)
    print(f"[Migration] Public file written: {public_output_path}")

    # 写入本地映射文件
    local_data = {"local_mappings": local_mappings}
    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(local_data, f, ensure_ascii=False, indent=2)
    print(f"[Migration] Local mapping file written: {local_output_path}")

    stats = {
        "total_benches": len(benches),
        "local_upload_count": local_count,
        "public_benches": len(public_benches),
        "local_mappings": len(local_mappings),
        "input_path": input_path,
        "public_output_path": public_output_path,
        "local_output_path": local_output_path,
    }

    print(f"[Migration] Migration completed!")
    print(f"  Total benches: {stats['total_benches']}")
    print(f"  Local uploads: {stats['local_upload_count']}")

    return stats


def main():
    """主函数"""
    # 确定路径
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    bench_table_dir = repo_root / "one_eval" / "utils" / "bench_table"

    input_path = bench_table_dir / "bench_gallery.json"
    public_output_path = bench_table_dir / "bench_gallery_public.json"
    local_output_path = bench_table_dir / "bench_gallery_local.json"

    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return 1

    # 检查输出文件是否已存在
    if public_output_path.exists() or local_output_path.exists():
        print(f"[Warning] Output files already exist!")
        print(f"  {public_output_path}: {'exists' if public_output_path.exists() else 'not exists'}")
        print(f"  {local_output_path}: {'exists' if local_output_path.exists() else 'not exists'}")

        response = input("Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("[Migration] Aborted by user")
            return 1

    # 执行迁移
    stats = migrate_bench_gallery(
        str(input_path),
        str(public_output_path),
        str(local_output_path),
        backup=True
    )

    # 提示更新 .gitignore
    print("\n[Next Steps]")
    print("1. Add the following to .gitignore:")
    print("   one_eval/utils/bench_table/bench_gallery_local.json")
    print("2. Update app.py to use the new file paths")
    print("3. Test the migration by running the server")

    return 0


if __name__ == "__main__":
    exit(main())

"""下载 HAI 21.03 并核对：文件大小与 git blob sha 均以 GitHub contents API 的返回为准，不一致则重下。

本地与集群通用（集群登录节点能上网时直接运行；否则从本地 rsync 后用 --verify-only 核对）。
核对结果写入数据目录下的 download_manifest.json。

用法：
    python -m scripts.download_hai [--verify-only] [--config config/config.yaml]
    python -m scripts.download_hai --offline      # 不联网：按数据目录里已有的 download_manifest.json 核对
                                                  # （数据从本地 rsync 到集群、计算节点不能上网时用）
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.utils.paths import DEFAULT_CONFIG, hai_data_dir, load_config

CHUNK = 1 << 20


def git_blob_sha(path: Path) -> str:
    """git blob sha1 = sha1(b"blob <size>\\0" + 内容)，与 contents API 的 sha 字段一致。"""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, size: int, sha: str) -> str | None:
    """返回不一致的原因；一致返回 None。"""
    if not path.exists():
        return "文件不存在"
    if path.stat().st_size != size:
        return f"大小不一致（本地 {path.stat().st_size}，期望 {size}）"
    if git_blob_sha(path) != sha:
        return "sha 不一致"
    return None


def download(url: str, path: Path, timeout_s: int) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description="下载并核对 HAI 数据集")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--verify-only", action="store_true", help="只核对，不下载")
    p.add_argument("--offline", action="store_true", help="不联网，按已有的 download_manifest.json 核对（隐含 --verify-only）")
    p.add_argument("--retries", type=int, default=3, help="单个文件核对失败后的最大重下次数")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    cfg = load_config(args.config)
    hai = cfg["ml"]["hai"]
    data_dir = hai_data_dir(cfg)
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = hai["train_files"] + hai["test_files"]

    manifest_path = data_dir / hai["manifest"]
    if args.offline:
        args.verify_only = True
        if not manifest_path.exists():
            print(f"[错误] 离线核对需要 {manifest_path}（随数据一起从本地同步）", file=sys.stderr)
            return 2
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        remote = {f["name"]: {"size": f["size"], "sha": f["git_blob_sha"]} for f in old["files"]}
    else:
        resp = requests.get(hai["api_url"], timeout=args.timeout)
        resp.raise_for_status()
        remote = {e["name"]: e for e in resp.json()}
    missing = [n for n in wanted if n not in remote]
    if missing:
        print(f"[错误] 官方仓库中找不到：{missing}", file=sys.stderr)
        return 2

    records, failed = [], []
    for name in wanted:
        size, sha = remote[name]["size"], remote[name]["sha"]
        path = data_dir / name
        reason = check(path, size, sha)
        attempts = 0
        while reason and not args.verify_only and attempts < args.retries:
            attempts += 1
            print(f"[下载] {name}（{reason}，第 {attempts} 次）")
            download(f"{hai['raw_url']}/{name}", path, args.timeout)
            reason = check(path, size, sha)
        print(f"[{'失败' if reason else '一致'}] {name}  {size} 字节  {reason or sha}")
        if reason:
            failed.append(name)
        records.append({"name": name, "size": size, "git_blob_sha": sha, "verified": reason is None})

    if args.offline:  # 离线核对不改写 manifest（其中的期望值来自官方仓库）
        return 1 if failed else 0
    manifest = {"version": hai["version"], "source": hai["api_url"],
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "total_bytes": sum(r["size"] for r in records), "files": records}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[保存] {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

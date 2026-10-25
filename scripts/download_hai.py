"""Download and verify HAI 21.03: file sizes and git blob shas are checked against the GitHub contents API; mismatches are re-downloaded.

Works both locally and on the cluster (run directly if the cluster login node has internet; otherwise rsync from local and verify with --verify-only).
Verification results are written to download_manifest.json in the data directory.

Usage:
    python -m scripts.download_hai [--verify-only] [--config config/config.yaml]
    python -m scripts.download_hai --offline      # no network: verify against the download_manifest.json already in the data directory
                                                  # (for when data was rsynced to the cluster and compute nodes have no internet)
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
    """git blob sha1 = sha1(b"blob <size>\\0" + content), identical to the sha field of the contents API."""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, size: int, sha: str) -> str | None:
    """Return the reason for a mismatch; None if everything matches."""
    if not path.exists():
        return "file does not exist"
    if path.stat().st_size != size:
        return f"size mismatch (local {path.stat().st_size}, expected {size})"
    if git_blob_sha(path) != sha:
        return "sha mismatch"
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
    p = argparse.ArgumentParser(description="Download and verify the HAI dataset")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--verify-only", action="store_true", help="only verify, do not download")
    p.add_argument("--offline", action="store_true", help="no network: verify against the existing download_manifest.json (implies --verify-only)")
    p.add_argument("--retries", type=int, default=3, help="maximum re-downloads per file after a failed verification")
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
            print(f"[ERROR] offline verification needs {manifest_path} (sync it from local together with the data)", file=sys.stderr)
            return 2
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        remote = {f["name"]: {"size": f["size"], "sha": f["git_blob_sha"]} for f in old["files"]}
    else:
        resp = requests.get(hai["api_url"], timeout=args.timeout)
        resp.raise_for_status()
        remote = {e["name"]: e for e in resp.json()}
    missing = [n for n in wanted if n not in remote]
    if missing:
        print(f"[ERROR] not found in the official repository: {missing}", file=sys.stderr)
        return 2

    records, failed = [], []
    for name in wanted:
        size, sha = remote[name]["size"], remote[name]["sha"]
        path = data_dir / name
        reason = check(path, size, sha)
        attempts = 0
        while reason and not args.verify_only and attempts < args.retries:
            attempts += 1
            print(f"[DOWNLOAD] {name} ({reason}, attempt {attempts})")
            download(f"{hai['raw_url']}/{name}", path, args.timeout)
            reason = check(path, size, sha)
        print(f"[{'FAIL' if reason else 'OK'}] {name}  {size} bytes  {reason or sha}")
        if reason:
            failed.append(name)
        records.append({"name": name, "size": size, "git_blob_sha": sha, "verified": reason is None})

    if args.offline:  # Offline verification does not rewrite the manifest (its expected values come from the official repository)
        return 1 if failed else 0
    manifest = {"version": hai["version"], "source": hai["api_url"],
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "total_bytes": sum(r["size"] for r in records), "files": records}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVED] {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared by the report scripts: where artifacts are read from, where results are written, which runs may be used.

Two modes that are never mixed:
- Official (default): reads runs under the artifact root, accepts only full runs on the cluster (not smoke, not local diagnostic, git_dirty=false, completed), writes to ml.report_dir.
- Smoke (--smoke): reads only smoke runs under <artifact root>/smoke/ and writes to <artifact root>/smoke_reports/. Only for development and pipeline checks;
  the numbers must not go into the report; in this mode reports/main/ and data/processed/ are never written.
"""

import argparse
import json
from pathlib import Path

from src.utils.paths import artifact_dir, resolve
from src.utils.run import MANIFEST_NAME


def report_problem(run_dir: Path, ok_status: tuple[str, ...] = ("completed",)) -> str | None:
    """Reason a run cannot be used for the official report; None if it can."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        return "no run_manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("local_diagnostic"):
        return "local diagnostic run (local_diagnostic=true, only for decisions, must not go into the report)"
    if m.get("smoke"):
        return "smoke-test artifacts (the numbers must not go into the report)"
    if m.get("git_dirty") is not False:
        return f"git_dirty = {m.get('git_dirty')} (must be false)"
    if m.get("status") not in ok_status:
        return f"status = {m.get('status')}"
    return None


def smoke_problem(run_dir: Path) -> str | None:
    """Reason a run cannot be used in smoke mode: it must really be a completed smoke run."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        return "no run_manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    if not m.get("smoke"):
        return "not a smoke run (smoke mode reads only smoke artifacts)"
    if m.get("status") != "completed":
        return f"status = {m.get('status')}"
    return None


class ReportSource:
    def __init__(self, cfg: dict, smoke: bool) -> None:
        self.cfg, self.smoke = cfg, smoke
        base = artifact_dir(cfg)
        sm = cfg["ml"]["smoke"]
        self.root = base / sm["artifact_subdir"] if smoke else base
        self.out = base / sm["report_subdir"] if smoke else resolve(cfg["ml"]["report_dir"])
        self.out.mkdir(parents=True, exist_ok=True)

    def problem(self, run_name: str) -> str | None:
        run_dir = self.root / run_name
        return smoke_problem(run_dir) if self.smoke else report_problem(run_dir)

    def usable_runs(self, run_names: list[str], required: tuple[str, ...] = ()) -> list[str]:
        """Names of the usable runs; unusable ones are skipped with the reason printed. required lists the artifact files that must exist."""
        ok = []
        for name in run_names:
            run_dir = self.root / name
            missing = [f for f in required if not (run_dir / f).exists()]
            problem = self.problem(name) if run_dir.exists() else "run directory does not exist (not run or not synced yet)"
            if not problem and missing:
                problem = f"missing {', '.join(missing)}"
            if problem:
                print(f"[SKIPPED] {name}: {problem}")
            else:
                ok.append(name)
        return ok

    def banner(self) -> None:
        if self.smoke:
            print(f"[SMOKE] reading {self.root}, writing to {self.out}; the numbers are only for checking the pipeline and must not go into the report")


def add_common_args(p: argparse.ArgumentParser) -> None:
    from src.utils.paths import DEFAULT_CONFIG
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--smoke", action="store_true",
                   help="develop and check the pipeline on smoke artifacts (<artifact dir>/smoke/), writing to <artifact dir>/smoke_reports/, not the report directory")

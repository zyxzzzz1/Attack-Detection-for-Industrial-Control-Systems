"""资产图：读取并校验 asset_map.yaml，提供 传感器 → 子系统 → 资产 → 入侵路径 的查询（claude.md M6）。

定义：
- 传感器 → 子系统：按 sensor_prefix 匹配（取最长前缀），未匹配返回 None。
- 上游资产：子系统 assets 列表中的资产，加上从它们沿 edges 反向可达的全部资产。
- 入侵路径：终点为子系统内的控制器（purdue_level == 1），起点为没有入边的资产，
  沿 edges 的全部极大简单路径，方向为 起点 → … → 控制器。

用法：
    python -m src.fusion.asset_graph [--asset-map YAML]
"""

import argparse
import sys
from ipaddress import IPv4Address
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]


class Subsystem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    criticality: float = Field(ge=0, le=1)
    sensor_prefix: str = Field(min_length=1)
    assets: list[str]
    note: str | None = None


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    purdue_level: int = Field(ge=0, le=5)
    ip: IPv4Address | None = None
    scanned: bool

    @model_validator(mode="after")
    def _scanned_needs_ip(self):
        if self.scanned and self.ip is None:
            raise ValueError("scanned: true 的资产必须有 ip")
        return self


class AssetMap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subsystems: dict[str, Subsystem]
    assets: dict[str, Asset]
    edges: list[tuple[str, str]]

    @model_validator(mode="after")
    def _check_refs(self):
        errors = []
        for sid, sub in self.subsystems.items():
            for a in sub.assets:
                if a not in self.assets:
                    errors.append(f"子系统 {sid} 引用了未定义的资产 {a}")
        for src, dst in self.edges:
            for end in (src, dst):
                if end not in self.assets:
                    errors.append(f"edge [{src}, {dst}] 的端点 {end} 未定义")
            if src == dst:
                errors.append(f"edge [{src}, {dst}] 是自环")
        prefixes = [s.sensor_prefix for s in self.subsystems.values()]
        if len(prefixes) != len(set(prefixes)):
            errors.append(f"sensor_prefix 有重复：{prefixes}")
        ips = [str(a.ip) for a in self.assets.values() if a.ip is not None]
        if len(ips) != len(set(ips)):
            errors.append(f"资产 ip 有重复：{ips}")
        if errors:
            raise ValueError("；".join(errors))
        return self


class AssetGraph:
    def __init__(self, amap: AssetMap):
        self.map = amap
        self.parents: dict[str, list[str]] = {a: [] for a in amap.assets}
        for src, dst in amap.edges:
            self.parents[dst].append(src)
        self._ip_to_asset = {str(a.ip): aid for aid, a in amap.assets.items() if a.ip is not None}

    @classmethod
    def from_yaml(cls, path: Path) -> "AssetGraph":
        with open(path, encoding="utf-8") as f:
            return cls(AssetMap.model_validate(yaml.safe_load(f)))

    @property
    def subsystems(self) -> dict[str, Subsystem]:
        return self.map.subsystems

    @property
    def assets(self) -> dict[str, Asset]:
        return self.map.assets

    def asset_by_ip(self, ip: str) -> str | None:
        return self._ip_to_asset.get(ip)

    def sensor_to_subsystem(self, sensor: str) -> str | None:
        """按最长 sensor_prefix 匹配；未知前缀返回 None。"""
        hits = [(len(s.sensor_prefix), sid) for sid, s in self.subsystems.items()
                if sensor.startswith(s.sensor_prefix)]
        return max(hits)[1] if hits else None

    def upstream_assets(self, sub_id: str) -> list[str]:
        """子系统的 assets 加上沿 edges 反向可达的全部资产（排序后返回）。"""
        seen: set[str] = set()
        stack = list(self.subsystems[sub_id].assets)
        while stack:
            a = stack.pop()
            if a not in seen:
                seen.add(a)
                stack.extend(self.parents[a])
        return sorted(seen)

    def intrusion_paths(self, sub_id: str) -> list[list[str]]:
        """以子系统内控制器为终点、以无入边资产为起点的全部极大简单路径。"""
        targets = [a for a in self.subsystems[sub_id].assets if self.assets[a].purdue_level == 1]
        paths: list[list[str]] = []

        def walk(node: str, rev_path: list[str]) -> None:
            # 无入边即为起点；有环时父节点全在路径上也视为终止，保证简单路径
            nxt = [p for p in self.parents[node] if p not in rev_path]
            if not nxt:
                paths.append(rev_path[::-1])
            for p in nxt:
                walk(p, rev_path + [p])

        for t in targets:
            walk(t, [t])
        return sorted(paths)


def main() -> int:
    p = argparse.ArgumentParser(description="校验 asset_map.yaml 并打印资产图")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--asset-map", type=Path, help="默认取 config.yaml 的 fusion.asset_map")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["fusion"]
    g = AssetGraph.from_yaml(args.asset_map or ROOT / cfg["asset_map"])
    for sid, sub in g.subsystems.items():
        print(f"{sid} {sub.name} criticality={sub.criticality}")
        print(f"  上游资产：{g.upstream_assets(sid)}")
        for path in g.intrusion_paths(sid):
            print(f"  路径：{' → '.join(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

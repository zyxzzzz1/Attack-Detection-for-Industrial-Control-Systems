"""Asset graph: read and validate asset_map.yaml and answer sensor -> subsystem -> asset -> intrusion path queries (claude.md M6).

Definitions:
- Sensor -> subsystem: matched by sensor_prefix (longest prefix wins); None if nothing matches.
- Upstream assets: the assets in the subsystem's assets list, plus every asset reachable from them backwards along edges.
- Intrusion paths: all maximal simple paths along edges that end at a controller of the subsystem (purdue_level == 1)
  and start at an asset with no incoming edge, directed start -> ... -> controller.

Usage:
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
            raise ValueError("assets with scanned: true must have an ip")
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
                    errors.append(f"subsystem {sid} references undefined asset {a}")
        for src, dst in self.edges:
            for end in (src, dst):
                if end not in self.assets:
                    errors.append(f"endpoint {end} of edge [{src}, {dst}] is undefined")
            if src == dst:
                errors.append(f"edge [{src}, {dst}] is a self-loop")
        prefixes = [s.sensor_prefix for s in self.subsystems.values()]
        if len(prefixes) != len(set(prefixes)):
            errors.append(f"duplicate sensor_prefix: {prefixes}")
        ips = [str(a.ip) for a in self.assets.values() if a.ip is not None]
        if len(ips) != len(set(ips)):
            errors.append(f"duplicate asset ip: {ips}")
        if errors:
            raise ValueError("; ".join(errors))
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
        """Match by the longest sensor_prefix; unknown prefixes return None."""
        hits = [(len(s.sensor_prefix), sid) for sid, s in self.subsystems.items()
                if sensor.startswith(s.sensor_prefix)]
        return max(hits)[1] if hits else None

    def upstream_assets(self, sub_id: str) -> list[str]:
        """The subsystem's assets plus all assets reachable backwards along edges (returned sorted)."""
        seen: set[str] = set()
        stack = list(self.subsystems[sub_id].assets)
        while stack:
            a = stack.pop()
            if a not in seen:
                seen.add(a)
                stack.extend(self.parents[a])
        return sorted(seen)

    def intrusion_paths(self, sub_id: str) -> list[list[str]]:
        """All maximal simple paths ending at a controller of the subsystem and starting at an asset with no incoming edge."""
        targets = [a for a in self.subsystems[sub_id].assets if self.assets[a].purdue_level == 1]
        paths: list[list[str]] = []

        def walk(node: str, rev_path: list[str]) -> None:
            # No incoming edge means a start; with a cycle, a node whose parents are all already on the path also ends the walk, keeping paths simple
            nxt = [p for p in self.parents[node] if p not in rev_path]
            if not nxt:
                paths.append(rev_path[::-1])
            for p in nxt:
                walk(p, rev_path + [p])

        for t in targets:
            walk(t, [t])
        return sorted(paths)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate asset_map.yaml and print the asset graph")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--asset-map", type=Path, help="defaults to fusion.asset_map in config.yaml")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["fusion"]
    g = AssetGraph.from_yaml(args.asset_map or ROOT / cfg["asset_map"])
    for sid, sub in g.subsystems.items():
        print(f"{sid} {sub.name} criticality={sub.criticality}")
        print(f"  upstream assets: {g.upstream_assets(sid)}")
        for path in g.intrusion_paths(sid):
            print(f"  path: {' → '.join(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Read-only GMP connectivity check.

Connects to gvmd via UnixSocketConnection and prints the GMP version, feed status, and the lists of
scan configs, scanners and port lists, plus the number of existing targets / tasks.
Creates, modifies or starts nothing.

User name and password are read only from the environment variables GVM_USER / GVM_PASSWORD;
socket path precedence: --socket > environment variable GVM_SOCKET > gvm.socket_path in config/config.yaml.

Usage:
    python -m scripts.check_gmp [--config config/config.yaml] [--socket PATH]
Exit codes: 0 all good; 1 connection or authentication failed; 2 missing credentials (only the version probe ran); 3 a key check failed
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from gvm.connections import UnixSocketConnection
from gvm.errors import GvmError
from gvm.protocols.gmp import Gmp
from gvm.transforms import EtreeCheckCommandTransform

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
REQUIRED_CONFIG = "Full and fast"


def load_socket_path(config_path: Path) -> str | None:
    """Read gvm.socket_path from the config file; return None if the file does not exist."""
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("gvm", {}).get("socket_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读检查 gvmd 的 GMP 连接")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    parser.add_argument(
        "--socket",
        default=os.environ.get("GVM_SOCKET"),
        help="gvmd socket 路径（覆盖 GVM_SOCKET 与配置文件）",
    )
    parser.add_argument("--timeout", type=int, default=60, help="socket 超时秒数")
    return parser.parse_args()


def names(response, tag: str) -> list[tuple[str, str]]:
    """Extract a list of (name, id) from a get_xxx response."""
    return [(el.findtext("name", ""), el.get("id", "")) for el in response.findall(tag)]


def main() -> int:
    args = parse_args()
    user = os.environ.get("GVM_USER")
    password = os.environ.get("GVM_PASSWORD")

    args.socket = args.socket or load_socket_path(args.config)
    if not args.socket:
        print(f"[失败] 未指定 socket 路径，且 {args.config} 中没有 gvm.socket_path")
        return 1
    if not os.path.exists(args.socket):
        print(f"[失败] socket 不存在或当前用户无法访问：{args.socket}")
        return 1

    connection = UnixSocketConnection(path=args.socket, timeout=args.timeout)
    try:
        with Gmp(connection=connection, transform=EtreeCheckCommandTransform()) as gmp:
            # get_version needs no authentication and can be used purely to probe connectivity
            version = gmp.get_version().findtext("version")
            print(f"[通过] 已连接 {args.socket}，GMP 版本 {version}")

            if not user or not password:
                print("[警告] 未设置 GVM_USER / GVM_PASSWORD，跳过认证后的检查")
                return 2

            gmp.authenticate(user, password)
            print(f"[通过] 用户 {user} 认证成功")
            ok = True

            print("\n== Feed 状态")
            for feed in gmp.get_feeds().findall("feed"):
                syncing = feed.find("currently_syncing") is not None
                print(
                    f"  {feed.findtext('type', ''):10} version={feed.findtext('version', '')}"
                    f"{'  [同步中]' if syncing else ''}"
                )

            print("\n== Scan configs")
            configs = names(gmp.get_scan_configs(filter_string="rows=-1"), "config")
            for name, uid in configs:
                print(f"  {name}  ({uid})")
            if not any(name == REQUIRED_CONFIG for name, _ in configs):
                print(f"[失败] 未找到 scan config “{REQUIRED_CONFIG}”")
                ok = False

            print("\n== Scanners")
            for name, uid in names(gmp.get_scanners(filter_string="rows=-1"), "scanner"):
                print(f"  {name}  ({uid})")

            print("\n== Port lists")
            port_lists = names(gmp.get_port_lists(filter_string="rows=-1"), "port_list")
            for name, uid in port_lists:
                print(f"  {name}  ({uid})")
            if not port_lists:
                print("[失败] port list 为空")
                ok = False

            targets = gmp.get_targets(filter_string="rows=-1").findall("target")
            tasks = gmp.get_tasks(filter_string="rows=-1").findall("task")
            print(f"\n== 已有 target 数量：{len(targets)}，task 数量：{len(tasks)}")

            return 0 if ok else 3
    except GvmError as e:
        print(f"[失败] GMP 错误：{e}")
        return 1
    except OSError as e:
        print(f"[失败] 连接 socket 出错：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

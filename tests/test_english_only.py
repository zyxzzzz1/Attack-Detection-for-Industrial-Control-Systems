"""Code, templates and config are English only: no Chinese characters in any tracked source file."""

import re
import subprocess

from src.utils.paths import ROOT

CODE_DIRS = ("src/", "scripts/", "app/", "templates/", "config/", "cluster/", "tests/", "env/")
CJK = re.compile("[%s-%s%s-%s%s-%s]" % tuple(map(chr, (0x3000, 0x303F, 0x4E00, 0x9FFF, 0xFF00, 0xFFEF))))


def test_no_chinese_in_code():
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                    cwd=ROOT, text=True).split()
    bad = []
    for f in files:
        path = ROOT / f
        if not f.startswith(CODE_DIRS) or path.suffix == ".md" or not path.is_file():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CJK.search(line):
                bad.append(f"{f}:{n}: {line.strip()[:80]}")
    assert bad == []

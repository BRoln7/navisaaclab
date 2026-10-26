"""Load a pinned checkout without installing MimicKit's generic top-level packages."""

import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def enable_mimickit():
    lock = json.loads((PROJECT_ROOT / "third_party/mimickit.lock.json").read_text())
    checkout = PROJECT_ROOT / "third_party/MimicKit"
    if not (checkout / "mimickit/learning/amp_agent.py").is_file():
        raise RuntimeError("Run scripts/mimickit_amp/fetch.py to fetch the pinned MimicKit dependency")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != lock["commit"] or dirty:
        raise RuntimeError(f"MimicKit must be an unmodified checkout of {lock['commit']}")
    path = str(checkout / "mimickit")
    if path not in sys.path:
        sys.path.insert(0, path)
    return lock

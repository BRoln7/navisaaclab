"""Fetch exactly the reviewed upstream revision; never change an existing checkout."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
lock = json.loads((root / "third_party/mimickit.lock.json").read_text())
checkout = root / "third_party/MimicKit"
if not checkout.exists():
    subprocess.run(["git", "clone", "--no-checkout", lock["repository"], str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", lock["commit"]], check=True)
actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
if actual != lock["commit"]:
    raise SystemExit(f"Existing checkout is {actual}; expected {lock['commit']}. It was left unchanged.")
print(f"MimicKit ready: {actual}")

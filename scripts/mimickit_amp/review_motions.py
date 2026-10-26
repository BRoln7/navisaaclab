"""Audit every native reference clip on CPU and plan four-view inspection batches."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_file", type=Path, default=ROOT / "data/mimickit/amass-pacer-v1/native/motions.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/motion-review/amass-pacer-v1")
    parser.add_argument("--max_seconds", type=float, default=8.)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max_seconds must be positive")
    from socnav_sim.mimickit_amp.motion_review import audit
    print(json.dumps(audit(args.motion_file, args.output, args.max_seconds), indent=2))


if __name__ == "__main__":
    main()

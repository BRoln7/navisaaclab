"""CPU-only composition audit and native MimicKit running oversampling list."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.sampling import build_running_sampling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/mimickit/official_smpl/locomotion/motions.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--running_weight", type=float, default=6.)
    args = parser.parse_args()
    output = args.output or args.source.with_name(f"motions-run{args.running_weight:g}.yaml")
    report = build_running_sampling(args.source, output, args.running_weight)
    print(json.dumps(report, indent=2))
    import torch
    assert not torch.cuda.is_initialized(), "Sampling preparation must remain CPU-only"


if __name__ == "__main__":
    main()

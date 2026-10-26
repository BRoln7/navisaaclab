"""Prepare curated walk/run sampling on CPU; does not run training or playback."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.bootstrap import enable_mimickit
from socnav_sim.mimickit_amp.curation import build_curated_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"data/cfg/mimickit_amp/motion_curation.yaml")
    parser.add_argument("--output", type=Path, help="New immutable output directory, overriding the configured default")
    args = parser.parse_args()
    enable_mimickit()
    report = build_curated_dataset(args.config, args.output)
    print(json.dumps({key: report[key] for key in ("groups", "frames", "duration_s", "history_sampling_share")}, indent=2))
    import torch
    assert not torch.cuda.is_initialized(), "Curation must not initialize CUDA"


if __name__ == "__main__":
    main()

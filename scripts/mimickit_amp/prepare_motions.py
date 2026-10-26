"""Prepare trusted local SMPL data on CPU. No Isaac Sim or training imports."""
import argparse
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "exts/socnav_sim"))
from socnav_sim.mimickit_amp.bootstrap import enable_mimickit
enable_mimickit()
from socnav_sim.mimickit_amp.data import convert
from socnav_sim.mimickit_amp.config import load
from socnav_sim.mimickit_amp.sampling import build_running_sampling
from socnav_sim.mimickit_amp.curation import build_curated_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--skeleton", type=Path, help="Defaults to the skeleton in the new feature's env config")
args = parser.parse_args()
if bool(args.source) != bool(args.output):
    parser.error("--source and --output must be supplied together")
settings = load("data/cfg/mimickit_amp/env.yaml")
skeleton = args.skeleton or root / settings["char_file"]
if not skeleton.is_file():
    parser.error(f"Missing official SMPL skeleton: {skeleton}. Run scripts/mimickit_amp/install_assets.py with the official data ZIP first.")
pairs = [(args.source, args.output)] if args.source else [
    (root / "data/motion_cap/amass_isaac/amass_isaac_run_upright_slim.pkl", root / "data/mimickit/official_smpl/locomotion"),
    (root / "data/motion_cap/amass_isaac/amass_isaac_standing_upright_slim.pkl", root / "data/mimickit/official_smpl/standing")]
for source, output in pairs:
    report = convert(source, skeleton, output)
    print(f"{output}: {len(report['clips'])} clips, {report['frames']} frames; skipped {len(report['skipped_short_clips'])} short clips", flush=True)
if not args.source:
    folder = root / "data/mimickit/official_smpl/locomotion"
    sampling = build_running_sampling(folder / "motions.yaml", folder / "motions-run6.yaml", running_weight=6.)
    print(f"Running sampling: {len(sampling['boosted_clips'])} clips with weight 6; {sampling['output']}", flush=True)
    curated = build_curated_dataset(root / "data/cfg/mimickit_amp/motion_curation.yaml")
    print(f"Curated locomotion: {len(curated['clips'])} clips; group shares "
          f"{ {group: info['sampling_share'] for group, info in curated['groups'].items()} }", flush=True)
import torch
assert not torch.cuda.is_initialized(), "Offline conversion must not initialize CUDA"

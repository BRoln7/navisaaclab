"""Run PACER preprocessing on downloaded AMASS, then convert to native MimicKit motions."""

import argparse
import json
import os
from pathlib import Path
import sys

# This entry point deliberately has no simulator/GPU path.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exts/socnav_sim"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, default=ROOT / "data/amass_raw/pacer_smplh_g/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/mimickit/amass-pacer-v1")
    parser.add_argument("--smpl_models", type=Path, default=ROOT / "data/smpl")
    parser.add_argument("--pacer_skeleton", type=Path, default=ROOT / "data/cfg/smpl_skeletontree.xml")
    parser.add_argument("--mimickit_skeleton", type=Path, default=ROOT / "third_party/MimicKit/data/assets/smpl/smpl.xml")
    parser.add_argument("--occlusion", type=Path,
                        help="Existing official annotation pickle; default: fetch published v2 from PACER download_data.sh")
    parser.add_argument("--time_mode", choices=("exact", "pacer_legacy"), default="exact",
                        help="exact: true 30 Hz; pacer_legacy: original integer stride and fixed 30 FPS label")
    parser.add_argument("--plan", action="store_true", help="Check source coverage/FPS without processing or downloading")
    args = parser.parse_args()
    if args.occlusion is not None and not args.occlusion.expanduser().is_file():
        parser.error("An explicit --occlusion must point to an existing annotation file")
    if args.occlusion is None:
        args.occlusion = ROOT / "data/amass_raw/pacer_smplh_g/support/amass_copycat_occlusion_v2.pkl"
    for name in ("raw_dir", "output", "smpl_models", "pacer_skeleton", "mimickit_skeleton", "occlusion"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    from socnav_sim.mimickit_amp.amass_processing import prepare, source_index
    indexed = source_index(args.raw_dir)
    print(f"[PACER] Found all {len(indexed)} unique selections; time mode: {args.time_mode}", flush=True)
    if args.plan:
        import numpy as np
        from collections import Counter
        counts = Counter()
        for path in indexed.values():
            with np.load(path, allow_pickle=False) as clip:
                counts[float(clip["mocap_framerate"].item())] += 1
        print(json.dumps({"source_fps_clip_counts": dict(counts), "output": str(args.output),
                          "occlusion_present": args.occlusion.is_file()}, indent=2))
        return
    report = prepare(args.raw_dir, args.output, args.smpl_models, args.pacer_skeleton,
                     args.mimickit_skeleton, args.occlusion, args.time_mode)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"[PACER] Ready for reference-motion inspection: {args.output / 'native/motions.yaml'}", flush=True)


if __name__ == "__main__":
    main()

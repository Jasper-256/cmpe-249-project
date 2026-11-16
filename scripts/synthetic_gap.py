"""Measure existing checkpoints on synthetic stand-ins for the real-domain gaps.

For each run directory, evaluates the selected checkpoint on synthetic
validation under (a) in-plane roll only and (b) the full sim2real-v1
augmentation, for both protocols. Synthetic data only. This tests, without
real data, whether models trained without roll fail when roll is added.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.sim2real import Sim2RealConfig, roll_only_config
from saga_pose.training.engine import ModelSpec, Protocol, build_model, evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sim2real/baseline_gap.json"))
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    results: dict[str, Any] = (
        json.loads(args.output.read_text(encoding="utf-8"))["runs"] if args.output.is_file() else {}
    )
    conditions = {"roll": roll_only_config(), "sim2real": Sim2RealConfig()}
    for run_dir in args.runs:
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        name = report["name"]
        if name in results:
            continue
        (object_name,) = report["config"]["objects"]
        validation = FreshCropCacheDataset(args.cache_dir / f"{object_name}-validation")
        loader: DataLoader[Any] = DataLoader(validation, batch_size=32, shuffle=False)
        spec = ModelSpec(**report["spec"])
        state = torch.load(
            Path(report["provenance"]["checkpoint"]), map_location=device, weights_only=True
        )
        model = build_model(spec, int(validation[0]["keypoints_xy"].shape[0]), pretrained=False)
        model.load_state_dict(state["model"])
        model.to(device)
        entry: dict[str, Any] = {"object": object_name, "trained_protocol": report["protocol"]}
        for protocol in ("natural", "tss"):
            for label, config in (("plain", None), *conditions.items()):
                result = evaluate(
                    model, loader, spec, cast(Protocol, protocol), device, augment=config
                )
                pose = result["pose"]["all_frames"]
                entry[f"{protocol}_{label}"] = {
                    "frames": result["frame_count"],
                    "rmse_px": result["coordinate_rmse_px"],
                    "are_deg": pose["are_degrees_mean"],
                    "ate_m": pose["absolute_translation_error_m_mean"],
                    "success_joint": pose["success_joint_10_deg_10_mm"],
                }
        results[name] = entry
        print(
            f"[{name}] natural ARE plain {entry['natural_plain']['are_deg']:.1f} "
            f"roll {entry['natural_roll']['are_deg']:.1f} "
            f"sim2real {entry['natural_sim2real']['are_deg']:.1f}",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "producer": "scripts/synthetic_gap.py",
                    "git_commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
                    ).stdout.strip(),
                    "device": torch.cuda.get_device_name(device),
                    "augmentations": {k: v.as_dict() for k, v in conditions.items()},
                    "evaluation_seed": 249,
                    "runs": results,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

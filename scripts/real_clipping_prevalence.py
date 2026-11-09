"""Pre-registered label-free descriptive analysis: how often real RGB crops clip.

Reads only the input clipping grids of the real caches (no labels, predictions or
errors) and reports the share of crops with any 16x16 patch at s >= 0.01, overall,
per object, and on the author harsh slice. Run only after the freeze (G6): it
says whether the real test can exercise the clipping signal at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.real import load_real_slices, slice_mask

OBJECTS = ("cassini", "satty", "soho")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/real/clipping_prevalence.json")
    )
    args = parser.parse_args()
    if not Path("outputs/real/freeze_manifest.json").is_file():
        raise SystemExit("run only after the freeze")
    with open_fresh_archive("real", args.raw_root) as archive:
        slices = load_real_slices(archive)
    report: dict[str, dict[str, float]] = {}
    all_active: list[np.ndarray] = []
    all_harsh: list[np.ndarray] = []
    for object_name in OBJECTS:
        cache = FreshCropCacheDataset(args.cache_dir / f"real-{object_name}")
        grid = np.asarray(cache.array("clipping_ratio")).reshape(len(cache), -1)
        active = (grid >= 0.01).any(axis=1)
        frames = np.asarray(cache.array("frame_number"))
        sequences = np.asarray(cache.sequences)
        harsh = np.zeros(len(cache), dtype=bool)
        for sequence, ranges in slices.items():
            harsh |= (sequences == sequence) & slice_mask(frames, ranges["harsh"])
        report[object_name] = {
            "crops": len(cache),
            "crops_with_active_patch": float(active.mean()),
            "harsh_crops_with_active_patch": float(active[harsh].mean()) if harsh.any() else 0.0,
            "median_active_patch_fraction_when_active": float(
                np.median((grid[active] >= 0.01).mean(axis=1)) if active.any() else 0.0
            ),
        }
        all_active.append(active)
        all_harsh.append(harsh)
    active_all, harsh_all = np.concatenate(all_active), np.concatenate(all_harsh)
    report["pooled"] = {
        "crops": int(active_all.size),
        "crops_with_active_patch": float(active_all.mean()),
        "harsh_crops_with_active_patch": float(active_all[harsh_all].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

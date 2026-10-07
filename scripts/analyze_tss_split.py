"""Split TSS-validation errors by whether each frame was actually corrupted.

Uses the per-frame predictions saved by ``train.py`` (own-protocol validation)
and the validation caches' ``tss_applied`` flags. Synthetic validation only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from saga_pose.data.cache import FreshCropCacheDataset

OBJECTS = ("cassini", "satty", "soho")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    applied: dict[str, bool] = {}
    for object_name in OBJECTS:
        cache = FreshCropCacheDataset(args.cache_dir / f"{object_name}-validation")
        flags = np.asarray(cache.array("tss_applied"), dtype=bool)
        applied.update(zip(cache.sample_ids, flags.tolist(), strict=True))
    rows: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    paths = sorted(
        args.run_root.glob("*/report.json"),
        key=lambda path: (
            OBJECTS.index(path.parent.name.rsplit("-", maxsplit=1)[1])
            if path.parent.name.rsplit("-", maxsplit=1)[1] in OBJECTS
            else len(OBJECTS)
        ),
    )
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["protocol"] != "tss" or report["outcome"]["stage"] != "train":
            continue
        frames = np.load(path.parent / "validation_frames.npz")
        method = report["name"].rsplit("-", maxsplit=1)[0]
        mask = np.asarray([applied[sample] for sample in frames["sample_id"]])
        for key in ("rotation_error_deg", "translation_error_m"):
            rows[method][f"{key}_corrupted"].append(frames[key][mask])
            rows[method][f"{key}_clean"].append(frames[key][~mask])
    lines = [
        "| method | object | corrupted ARE mean / median | corrupted ATE mean (mm) "
        "| clean ARE mean / median | clean ATE mean (mm) |",
        "|---|---|---|---|---|---|",
    ]
    for method in sorted(rows):
        parts = rows[method]
        if len(parts["rotation_error_deg_corrupted"]) != len(OBJECTS):
            continue
        groups = [("pooled", slice(None))] + [
            (object_name, slice(index, index + 1)) for index, object_name in enumerate(OBJECTS)
        ]
        for label, selection in groups:
            values = {key: np.concatenate(chunks[selection]) for key, chunks in parts.items()}
            corrupted = values["rotation_error_deg_corrupted"]
            clean = values["rotation_error_deg_clean"]
            lines.append(
                f"| {method} | {label} | {corrupted.mean():.2f} / {np.median(corrupted):.2f} "
                f"| {1000 * values['translation_error_m_corrupted'].mean():.1f} "
                f"| {clean.mean():.2f} / {np.median(clean):.2f} "
                f"| {1000 * values['translation_error_m_clean'].mean():.1f} |"
            )
    text = "\n".join(lines) + "\n"
    (args.output or args.run_root / "tss_split.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

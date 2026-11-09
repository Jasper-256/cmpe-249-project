"""Freeze every selected checkpoint before the one-time real evaluation (G6).

For each run declared in the given matrix configs, record the checkpoint SHA-256,
selected epoch, model spec, protocol, and seed. The freeze is refused if any
declared run lacks a completed synthetic report, so the real test cannot start
on a partial matrix. The manifest is written once; an existing manifest is never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrices", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/real/freeze_manifest.json"))
    args = parser.parse_args()
    if args.output.is_file():
        raise SystemExit(f"{args.output} exists; the freeze is written once")
    runs: dict[str, Any] = {}
    missing: list[str] = []
    for matrix_path in args.matrices:
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        output_dir = Path(matrix["defaults"]["output_dir"])
        for entry in matrix["runs"]:
            run_dir = output_dir / entry["name"]
            report_path = run_dir / "report.json"
            if not report_path.is_file():
                missing.append(entry["name"])
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            checkpoint = Path(report["provenance"]["checkpoint"])
            runs[entry["name"]] = {
                "run_dir": run_dir.as_posix(),
                "matrix": str(matrix_path),
                "spec": report["spec"],
                "protocol": report["protocol"],
                "seed": report["config"]["seed"],
                "selected_epoch": report["outcome"]["selected_epoch"],
                "selection_rule": "minimum synthetic-validation keypoint RMSE (own protocol)",
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": _sha256(checkpoint),
            }
    if missing:
        raise SystemExit(f"refusing to freeze: {len(missing)} declared runs incomplete: {missing}")

    def git(*command: str) -> str:
        return subprocess.run(
            ["git", *command], check=True, capture_output=True, text=True
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "src", "scripts", "configs"))
    manifest = {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "source_tree_dirty": dirty,
        "uv_lock_sha256": _sha256(Path("uv.lock")),
        "geometric_selection_sha256": _sha256(Path("outputs/geometric/selection.json")),
        "comparisons_sha256": _sha256(Path("configs/experiments/main_v1_comparisons.yaml")),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"froze {len(runs)} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

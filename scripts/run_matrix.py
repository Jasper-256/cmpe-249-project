"""Run a declared experiment matrix sequentially, resuming completed runs.

Each entry in the YAML ``runs`` list maps to ``scripts/train.py`` arguments;
``defaults`` apply to every entry. A run whose ``report.json`` exists is skipped,
so an interrupted matrix restarts where it stopped. Failures are recorded in the
matrix ledger and do not stop later runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--only", nargs="*", default=None, help="run names to include")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--shard",
        default=None,
        help="K/N: run only entries whose index modulo N equals K, with a per-shard "
        "ledger, so N runners can share one large GPU",
    )
    return parser.parse_args()


def _arguments(entry: dict[str, Any]) -> list[str]:
    arguments: list[str] = []
    for key, value in entry.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            arguments.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            arguments.extend([flag, *map(str, value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def _keep_system_awake() -> None:
    """Ask Windows not to idle-sleep while this runner lives (no settings change).

    On 2026-10-06 the laptop entered Modern Standby one minute after an
    overnight launch and froze training for 13 hours. The request lapses
    automatically when the process exits. Closing the lid can still force sleep.
    """
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        _keep_system_awake()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    defaults = matrix.get("defaults", {})
    output_dir = Path(defaults.get("output_dir", "outputs/runs"))
    shard_index, shard_count = 0, 1
    if args.shard:
        shard_index, shard_count = (int(part) for part in args.shard.split("/"))
    suffix = f"-shard{shard_index}of{shard_count}" if shard_count > 1 else ""
    ledger_path = output_dir / f"{args.matrix.stem}-ledger{suffix}.json"
    ledger: dict[str, Any] = (
        json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
    )
    for index, entry in enumerate(matrix["runs"]):
        if index % shard_count != shard_index:
            continue
        merged = {**defaults, **entry}
        merged.setdefault("checkpoint_dir", str(Path("checkpoints") / output_dir.name))
        name = merged["name"]
        if args.only and name not in args.only:
            continue
        report = output_dir / name / "report.json"
        if report.is_file():
            print(f"skip {name} (report exists)", flush=True)
            continue
        command = [sys.executable, "scripts/train.py", *_arguments(merged)]
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        log_path = Path(merged["checkpoint_dir"]) / name / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        # Append: a resumed run must keep the log of its earlier epochs.
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        ledger[name] = {
            "matrix": str(args.matrix),
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "log": str(log_path),
            "status": "complete" if report.is_file() else "failed",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        print(
            f"{name}: {ledger[name]['status']} in {ledger[name]['seconds'] / 60:.1f} min",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

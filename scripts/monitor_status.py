"""Condense run logs and reports into one small JSON status for the live monitor page.

Reads only local logs, synthetic run reports, and (after G6) real-test analysis
outputs. Writes ``outputs/monitor/status.json``; the published monitor page
renders this document.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

OBJECTS = ("cassini", "satty", "soho")
MATRICES = {
    "main": (Path("configs/experiments/main_v1.yaml"), Path("checkpoints/main-hrnet")),
    "finetune": (Path("configs/experiments/finetune_v1.yaml"), Path("checkpoints/finetune-hrnet")),
}
EPOCH = re.compile(
    r"\[(?P<name>[^\]]+)\] epoch (?P<e>\d+)/(?P<n>\d+) .*?"
    r"val_rmse=(?P<rmse>[\d.]+)px best=(?P<best>[\d.]+)@(?P<at>\d+) \((?P<sec>\d+)s\)"
)


def _matrix(config: Path, checkpoint_dir: Path) -> dict[str, Any]:
    matrix = yaml.safe_load(config.read_text(encoding="utf-8"))
    output_dir = Path(matrix["defaults"]["output_dir"])
    names = [run["name"] for run in matrix["runs"]]
    done = [n for n in names if (output_dir / n / "report.json").is_file()]
    ledger_path = output_dir / f"{config.stem}-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
    minutes = [v["seconds"] / 60 for v in ledger.values() if v.get("status") == "complete"]
    failed = [n for n, v in ledger.items() if v.get("status") == "failed" and n not in done]
    remaining = [n for n in names if n not in done]
    current: dict[str, Any] | None = None
    logs = sorted(
        (p for p in checkpoint_dir.glob("*/train.log") if p.parent.name in remaining),
        key=lambda p: p.stat().st_mtime,
    )
    if logs:
        log = logs[-1]
        matches = list(EPOCH.finditer(log.read_text(encoding="utf-8", errors="replace")))
        current = {"name": log.parent.name, "updated": _iso(log.stat().st_mtime)}
        if matches:
            m = matches[-1]
            current.update(
                epoch=int(m["e"]),
                epochs=int(m["n"]),
                val_rmse_px=float(m["rmse"]),
                best_px=float(m["best"]),
                best_epoch=int(m["at"]),
                epoch_seconds=int(m["sec"]),
            )
    recent = sorted(
        ((n, v) for n, v in ledger.items() if v.get("status") == "complete"),
        key=lambda item: (
            (output_dir / item[0] / "report.json").stat().st_mtime
            if (output_dir / item[0] / "report.json").is_file()
            else 0
        ),
    )[-6:]
    avg = sum(minutes[-20:]) / len(minutes[-20:]) if minutes else None
    return {
        "total": len(names),
        "done": len(done),
        "remaining": len(remaining),
        "failed_pending_retry": failed,
        "avg_run_minutes": round(avg, 1) if avg else None,
        "eta_hours": round(len(remaining) * avg / 60, 1) if avg else None,
        "current": current,
        "recent": [
            {"name": n, "minutes": round(v["seconds"] / 60, 1)} for n, v in reversed(recent)
        ],
        "next": remaining[1:4] if current else remaining[:3],
    }


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")


def _pooled(method: str, root: Path, key: str) -> dict[str, float] | None:
    total: dict[str, float] = {"are": 0.0, "ate": 0.0, "succ": 0.0}
    frames = 0
    for o in OBJECTS:
        path = root / f"{method}-{o}" / "report.json"
        if not path.is_file():
            return None
        evaluation = json.loads(path.read_text(encoding="utf-8"))["outcome"]["evaluations"].get(key)
        if evaluation is None:
            return None
        n = evaluation["frame_count"]
        pose = evaluation["pose"]["all_frames"]
        total["are"] += pose["are_degrees_mean"] * n
        total["ate"] += pose["absolute_translation_error_m_mean"] * 1000 * n
        total["succ"] += pose["success_joint_10_deg_10_mm"] * n
        frames += n
    return {k: round(v / frames, 3) for k, v in total.items()}


RESULT_ROWS = [
    ("RGB only", "u-rgb-natural", "natural", "main"),
    ("RGB only", "u-rgb-tss", "tss", "main"),
    ("Event only", "u-event-activity", "natural", "main"),
    ("Mean fusion", "f-mean-tss-activity", "tss", "main"),
    ("Content gate", "f-content_gate-tss-activity", "tss", "main"),
    ("Content + clipping gate", "f-content_clip_gate-tss-activity", "tss", "main"),
    ("Clipping gate", "f-clip_gate-tss-activity", "tss", "main"),
    ("Constant control", "f-clip_gate_constant-tss-activity", "tss", "main"),
    ("Fixed clipping gate", "f-clip_gate_fixed-tss-activity", "tss", "main"),
    ("Coarse attention", "f-attention-tss-activity", "tss", "main"),
    ("Attention + clipping bias", "f-attention_clip-tss-activity", "tss", "main"),
    ("Mean fusion, fine-tuned", "ft-mean-tss-activity", "tss", "finetune"),
    ("Concat fusion", "f-concat-natural-activity", "natural", "main"),
]


def main() -> int:
    roots = {
        "main": Path("outputs/runs/main-hrnet"),
        "finetune": Path("outputs/runs/finetune-hrnet"),
    }
    matrices = {k: _matrix(*v) for k, v in MATRICES.items()}
    results = []
    for label, method, protocol, root in RESULT_ROWS:
        natural = _pooled(method, roots[root], "validation_natural")
        tss = _pooled(method, roots[root], "validation_tss")
        if natural and tss:
            results.append(
                {
                    "label": label,
                    "method": method,
                    "trained": protocol,
                    "natural": natural,
                    "tss": tss,
                }
            )
    solver = Path("outputs/geometric/solver_control.json")
    g6_log = Path("checkpoints/g6.log")
    g6: dict[str, Any] = {"started": False}
    if g6_log.is_file():
        text = g6_log.read_text(encoding="utf-8", errors="replace")
        manifest = Path("outputs/real/freeze_manifest.json")
        total = (
            len(json.loads(manifest.read_text(encoding="utf-8"))["runs"])
            if manifest.is_file()
            else None
        )
        g6 = {
            "started": "G6 start" in text,
            "frozen": manifest.is_file(),
            "evaluated": len(list(Path("outputs/real").glob("*/real_report.json"))),
            "total": total,
            "failed": text.count("FAILED"),
            "done": "G6 done" in text,
        }
    primary = None
    analysis = Path("outputs/real/analysis_primary.json")
    if analysis.is_file():
        primary = json.loads(analysis.read_text(encoding="utf-8")).get("primary")
    runner = Path("checkpoints/main-hrnet/runner.log").read_text(encoding="utf-8", errors="replace")
    if g6.get("done"):
        stage = "Real-test analysis complete"
    elif g6.get("started"):
        stage = "Real-test evaluation (G6) running"
    elif "retry pass" in runner.split("overnight start")[-1]:
        stage = "Retrying failed runs"
    elif matrices["main"]["remaining"] or matrices["finetune"]["remaining"]:
        stage = "Training the experiment matrix"
    else:
        stage = "Training complete, waiting for G6"
    status = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": stage,
        "matrices": matrices,
        "results": results,
        "solver_control": json.loads(solver.read_text(encoding="utf-8"))
        if solver.is_file()
        else None,
        "g6": g6,
        "primary_real": primary,
    }
    out = Path("outputs/monitor/status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("stage", "generated_at")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

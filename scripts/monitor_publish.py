"""Build the monitor page and the live-feed document from the latest status.

Writes ``outputs/monitor/monitor.publish.html`` (the page with the current
status embedded as a fallback snapshot) and ``outputs/monitor/feed.json`` (the
``status/current`` database document the published page subscribes to).
Run ``scripts/monitor_status.py`` first.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("outputs/monitor")


def main() -> int:
    status = json.loads((ROOT / "status.json").read_text(encoding="utf-8"))
    status.pop("solver_control", None)  # not rendered; keeps the feed small
    payload = json.dumps(status, separators=(",", ":"))
    page = (ROOT / "monitor.html").read_text(encoding="utf-8")
    embedded = payload.replace("</", "<\\/")
    (ROOT / "monitor.publish.html").write_text(
        page.replace("__SNAPSHOT__", embedded), encoding="utf-8"
    )
    (ROOT / "feed.json").write_text(json.dumps({"payload": payload}), encoding="utf-8")
    print(f"payload {len(payload)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

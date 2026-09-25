#!/usr/bin/env python3
"""Print MONITOR.md finding lines for one progress.jsonl. Observes; does not grade."""

import json
import sys
from pathlib import Path

from localbench.render import monitor_findings


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: monitor-report.py <progress.jsonl>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for line in monitor_findings(events, path.parent.name):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

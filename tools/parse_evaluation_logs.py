#!/usr/bin/env python3
"""Parse Video-MoFlow LOSO evaluation logs into a single comparison CSV.

Reads the tail of every

    <logs_root>/<Model_Type>/<Scene_Name>/log.txt

extracts ``--Metric(Horizon): Value`` pairs via regex, and aggregates them
into one wide CSV with columns::

    Dataset, Metric, Horizon, No Video, Global VE

Log line format (multiple tab-separated pairs per INFO line are handled)::

    2026-08-23 21:53:18,091   INFO  --ADE_min(2.4s): 0.5762797	--FDE_min(2.0s): 0.5879701

Usage::

    python tools/parse_evaluation_logs.py                       # defaults
    python tools/parse_evaluation_logs.py --scenes gates quad --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_LOGS_ROOT = "/home/dhya/Documents/Studies/Thesis/results/MoFlow"
DEFAULT_OUT = "evaluation_results.csv"
DEFAULT_TAIL_LINES = 50

#: CSV column header -> directory name under ``logs_root``.
MODEL_COLUMNS: dict[str, str] = {
    "No Video": "No Video",   # trajectory-only baseline
    "Global VE": "Global",    # custom global video encoder
}

SCENES: tuple[str, ...] = (
    "bookstore", "coupa", "deathCircle", "gates",
    "hyang", "little", "nexus", "quad",
)

#: ``--ADE_min(2.4s): 0.5762797``  ->  ("ADE_min", "2.4s", "0.5762797")
METRIC_RE = re.compile(
    r"--(?P<metric>[A-Za-z]\w*)"      # metric name, e.g. ADE_min / JFDE_avg / MASD
    r"\((?P<horizon>[\d.]+s)\)"       # horizon tag, e.g. 1.2s / 4.8s
    r":\s*(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_log_file(log_path: Path, tail_lines: int) -> dict[tuple[str, str], float]:
    """Extract metrics from the last ``tail_lines`` lines of a log file.

    Returns ``{(metric, horizon): value}``. When a (metric, horizon) pair
    appears multiple times (repeated evaluation blocks), the **last**
    occurrence wins — i.e. the most recent evaluation in the window.
    """
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        tail = f.readlines()[-tail_lines:]

    metrics: dict[tuple[str, str], float] = {}
    for line in tail:
        for m in METRIC_RE.finditer(line):
            metrics[(m["metric"], m["horizon"])] = float(m["value"])
    return metrics


def _horizon_sort_key(horizon: str) -> float:
    """Numeric sort key for horizon tags such as '1.2s'."""
    return float(horizon.rstrip("s"))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def build_rows(
    logs_root: Path,
    scenes: list[str],
    tail_lines: int,
) -> tuple[list[list[str]], list[str]]:
    """Build wide-format CSV rows across scenes and models.

    Returns ``(rows, missing)`` where ``rows`` are already ordered and
    ``missing`` lists every expected log file that could not be parsed.
    """
    missing: list[str] = []
    # per scene: {model_column: {(metric, horizon): value}}
    per_scene: dict[str, dict[str, dict[tuple[str, str], float]]] = {}

    for scene in scenes:
        per_scene[scene] = {}
        for col_name, dir_name in MODEL_COLUMNS.items():
            log_path = logs_root / dir_name / scene / "log.txt"
            try:
                per_scene[scene][col_name] = parse_log_file(log_path, tail_lines)
            except FileNotFoundError:
                print(f"[warn] missing log: {log_path}", file=sys.stderr)
                missing.append(str(log_path))
                per_scene[scene][col_name] = {}

    rows: list[list[str]] = []
    for scene in scenes:
        # Union of keys across models so absent values become empty cells.
        key_set: set[tuple[str, str]] = set()
        for col in MODEL_COLUMNS:
            key_set.update(per_scene[scene].get(col, {}))
        for metric, horizon in sorted(
            key_set, key=lambda kh: (kh[0], _horizon_sort_key(kh[1]))
        ):
            row = [scene, metric, horizon]
            for col in MODEL_COLUMNS:
                val = per_scene[scene][col].get((metric, horizon))
                row.append(f"{val:.7f}" if val is not None else "")
            rows.append(row)
    return rows, missing


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate Video-MoFlow LOSO evaluation logs into one CSV.",
    )
    p.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT,
                   help="Root containing '<Model>/<Scene>/log.txt' trees.")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Output CSV path (default: %(default)s).")
    p.add_argument("--scenes", nargs="*", default=list(SCENES),
                   help="Scene subset (default: all SDD scenes).")
    p.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES,
                   help="Lines to read from each log end (default: %(default)s).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logs_root = Path(args.logs_root).expanduser()

    rows, missing = build_rows(logs_root, args.scenes, args.tail_lines)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Metric", "Horizon", *MODEL_COLUMNS])
        writer.writerows(rows)

    n_models = len(MODEL_COLUMNS)
    print(f"[parse] wrote {len(rows)} metric rows x {n_models + 3} columns -> {out_path}")
    if missing:
        print(f"[parse] {len(missing)} log file(s) missing; cells left empty.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Success rate vs checkpoint step, one line per task, from `*.summary.json`.

    python scripts/results_plot.py eval_result/baseline_finetuned-50k
    python scripts/results_plot.py eval_result/baseline_finetuned-50k --seed-offset 1
    python scripts/results_plot.py eval_result/baseline_finetuned-50k --output sr.png

Reads the same files as `scripts/results_table.py --pivot` and shares its
parsing *and its seed-block filter*, so the picture and the table cannot
disagree. `--seed-offset` defaults to 0, the reported block; the step-count
decision is made on 1.

The CI95 band is drawn by default and the mean is drawn heavier than the task
lines on purpose. At 100 episodes the interval is roughly +/-10 points, so a
per-task argmax is mostly noise; the step count has to be frozen once and
applied to every task (docs/BENCHMARK.md, "Choosing the step count"). The
dashed vertical marks the best *mean*, which is the choice this plot is for.

matplotlib is an optional dependency -- nothing else in this repo needs it:

    pip install --only-binary=:all: matplotlib
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from results_table import collect, is_partial, pivot_cells, select_seed_offset  # noqa: E402


def build_series(rows: list[dict]) -> tuple[list[str], list[int], dict]:
    """``(tasks, steps, cells)`` restricted to rows that have a success rate."""
    cells, _ = pivot_cells(rows)
    cells = {k: v for k, v in cells.items() if isinstance(v["sr_pct"], (int, float))}
    tasks = sorted({task for task, _ in cells})
    steps = sorted({step for _, step in cells})
    return tasks, steps, cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="directory to search for *.summary.json")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="plot only runs from this seed block (default 0, the "
                         "reported block; 1 is the checkpoint-selection sweep)")
    ap.add_argument("--pattern", default=None,
                    help="only include runs whose path matches this glob, "
                         "e.g. '*v2*' to keep only rerun summaries")
    ap.add_argument("--output", default="sr_vs_steps.png", help="PNG to write")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-ci", dest="ci", action="store_false",
                    help="omit the CI95 bands (they are the point; think twice)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    root = Path(args.results_dir).expanduser()
    rows = collect(root)
    if not rows:
        print(f"no *.summary.json under {root}", file=sys.stderr)
        return 1

    if args.pattern:
        rows = [r for r in rows if fnmatch.fnmatch(r["run"], args.pattern)]
        if not rows:
            print(f"no runs under {root} match pattern {args.pattern!r}", file=sys.stderr)
            return 1

    rows, unattributable = select_seed_offset(rows, args.seed_offset)
    if not rows:
        found = sorted({r["seed_offset"] for r in collect(root)
                        if r["seed_offset"] is not None})
        print(f"no runs under {root} are from seed offset {args.seed_offset}."
              + (f" Offsets present: {found}." if found else ""), file=sys.stderr)
        return 1
    if unattributable:
        print(f"note: {len(unattributable)} run(s) carry no seed<N> in their path "
              f"and were left out", file=sys.stderr)

    tasks, steps, cells = build_series(rows)
    if not cells:
        print("no summary carries both a task and a ckpt<N> step", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("error: results_plot.py needs matplotlib, which this repo does not "
              "otherwise require.\n"
              "       pip install --only-binary=:all: matplotlib\n"
              "       (or use `scripts/results_table.py --pivot`, standard library only)",
              file=sys.stderr)
        return 2

    fig, ax = plt.subplots(figsize=(8, 5))

    for task in tasks:
        points = [(s, cells[(task, s)]) for s in steps if (task, s) in cells]
        xs = [s for s, _ in points]
        ys = [r["sr_pct"] for _, r in points]
        line, = ax.plot(xs, ys, marker="o", linewidth=1.5, label=task)
        if args.ci:
            band = [(s, r["ci95_pct"]) for s, r in points if r["ci95_pct"]]
            if band:
                ax.fill_between([s for s, _ in band],
                                [ci[0] for _, ci in band],
                                [ci[1] for _, ci in band],
                                color=line.get_color(), alpha=0.12, linewidth=0)
        for s, r in points:
            if is_partial(r):
                ax.plot([s], [r["sr_pct"]], marker="x", color="black",
                        markersize=9, linestyle="none", zorder=5)

    # Mean only where every task reported, so it cannot move because the task
    # set changed between columns.
    mean_steps, mean_values = [], []
    for step in steps:
        values = [cells[(t, step)]["sr_pct"] for t in tasks if (t, step) in cells]
        if len(values) == len(tasks):
            mean_steps.append(step)
            mean_values.append(sum(values) / len(values))

    if mean_values:
        ax.plot(mean_steps, mean_values, marker="s", color="black",
                linewidth=2.5, linestyle="--", label=f"mean of {len(tasks)} tasks", zorder=4)
        best = mean_steps[mean_values.index(max(mean_values))]
        ax.axvline(best, color="black", linewidth=1, linestyle=":", alpha=0.6)
        # Pinned to the top of the axes rather than to the point, so it cannot
        # land on top of a task line.
        ax.annotate(f"best mean: {best:,}", xy=(best, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -13), textcoords="offset points", fontsize=9)

    ax.set_xlabel("finetuning step")
    ax.set_ylabel("success rate (%)")
    ax.set_title(args.title or f"UniVTAC success rate vs checkpoint  "
                               f"({root.name}, seed offset {args.seed_offset})")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s // 1000}k" for s in steps])
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = Path(args.output).expanduser()
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")
    if any(is_partial(r) for r in cells.values()):
        print("note: points marked x scored fewer episodes than requested")
    if mean_values:
        print(f"best mean across all {len(tasks)} tasks: step {best}"
              f" ({max(mean_values):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

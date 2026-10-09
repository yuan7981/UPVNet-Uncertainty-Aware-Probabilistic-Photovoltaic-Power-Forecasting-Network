"""
Recompute metrics from a saved run.

`scripts/train_upvnet.py` writes `test_predictions.npz` (true / predicted mean /
predicted variance). This script reloads that artifact and recomputes the full
metric set, including the probabilistic metrics, without retraining.

    python scripts/evaluate.py --run results/station01_seq96_pred1

Use `--scale physical --power-mean <kW> --power-std <kW>` to report in the
original power units rather than Z-score standardised ones (see
REPRODUCIBILITY.md, "Value scale by table").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.upvnet.metrics import compute_metrics, format_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute metrics from a saved run")
    ap.add_argument("--run", required=True, help="run directory containing test_predictions.npz")
    ap.add_argument(
        "--scale",
        choices=["standardized", "physical"],
        default="standardized",
        help="value scale to report in (default: standardized, i.e. Z-score)",
    )
    ap.add_argument("--power-mean", type=float, default=0.0, help="train-split power mean (for --scale physical)")
    ap.add_argument("--power-std", type=float, default=1.0, help="train-split power std (for --scale physical)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    npz_path = run_dir / "test_predictions.npz"
    if not npz_path.exists():
        print(f"error: {npz_path} not found", file=sys.stderr)
        return 1

    data = np.load(npz_path)
    true = data["true"].astype(np.float64)
    pred = data["pred"].astype(np.float64)
    pred_var = data["pred_var"].astype(np.float64)

    note = "standardized (Z-score) scale"
    if args.scale == "physical":
        true = true * args.power_std + args.power_mean
        pred = pred * args.power_std + args.power_mean
        pred_var = pred_var * (args.power_std ** 2)
        note = f"physical scale (mean={args.power_mean}, std={args.power_std})"

    metrics = compute_metrics(pred, true, pred_var)
    print(f"# {run_dir}  —  {note}")
    print(format_metrics(metrics))
    print()
    print(json.dumps(metrics, indent=2))

    out_path = run_dir / f"metrics_{args.scale}.json"
    out_path.write_text(
        json.dumps({"scale": args.scale, "power_mean": args.power_mean,
                    "power_std": args.power_std, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

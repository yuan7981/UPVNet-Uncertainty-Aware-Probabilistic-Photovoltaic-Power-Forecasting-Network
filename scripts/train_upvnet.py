"""
Train (and test) UPVNet on one dataset / prediction horizon.

Examples
--------
Reproduce the station01 H=1 result reported in the paper:

    python scripts/train_upvnet.py --dataset station01 --pred-len 1

All four horizons on both paper datasets:

    python scripts/train_upvnet.py --dataset station01      --pred-len 1
    python scripts/train_upvnet.py --dataset station01      --pred-len 4
    python scripts/train_upvnet.py --dataset station01      --pred-len 24
    python scripts/train_upvnet.py --dataset station01      --pred-len 96
    python scripts/train_upvnet.py --dataset 23-Calyxodataset --pred-len 1

Requires `preprocessed/<dataset>/{train,val,test}.csv`; produce them first
with `python -m src.preprocess.common_preprocess` (see the README).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.upvnet.train import TrainConfig, run_experiment  # noqa: E402

# Prediction horizons used in the paper (pred_len steps at 15-minute resolution).
HORIZONS = {"15min": 1, "1h": 4, "6h": 24, "24h": 96}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train UPVNet on a preprocessed dataset split")
    ap.add_argument("--dataset", default="station01", help="dataset folder name under preprocessed/")
    ap.add_argument(
        "--horizon",
        choices=sorted(HORIZONS),
        default=None,
        help="named horizon (15min/1h/6h/24h); overrides --pred-len",
    )
    ap.add_argument("--pred-len", type=int, default=1, help="prediction length in 15-min steps")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cnnrs-channels", type=int, default=64)
    ap.add_argument("--rgbf-hidden", type=int, default=64)
    ap.add_argument("--attn-dim", type=int, default=64)
    ap.add_argument("--data-root", default="preprocessed")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pred_len = HORIZONS[args.horizon] if args.horizon else args.pred_len
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_ROOT / "results" / f"{args.dataset}_seq{args.seq_len}_pred{pred_len}"
    )

    cfg = TrainConfig(
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=pred_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        cnnrs_channels=args.cnnrs_channels,
        rgbf_hidden=args.rgbf_hidden,
        attn_dim=args.attn_dim,
        data_root=args.data_root,
    )

    def logger(msg: str) -> None:
        print(msg, flush=True)

    summary = run_experiment(cfg, PROJECT_ROOT, out_dir, logger=logger)
    print("\n=== test metrics ===")
    print(json.dumps(summary["test_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Training loop for UPVNet.

The loss, schedule and optimiser settings here are the ones used for the
results reported in the paper:

    Loss = MSE(mu, y) + lambda * NLL(mu, sigma^2, y) + alpha * mean(sigma)

`lambda` is warmed up: the first `mse_pretrain_epochs` epochs train on MSE
only, then `lambda` ramps linearly to `nll_weight_max` over
`nll_warmup_epochs`. The `alpha * mean(sigma)` term is a sharpness penalty
that stops the predicted variance from collapsing to a degenerate wide
interval.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import create_dataloaders, load_csv_to_dataset
from .metrics import compute_metrics, format_metrics
from .model import UPVNet


@dataclass
class TrainConfig:
    """Everything needed to reproduce a single run."""

    dataset: str = "station01"
    seq_len: int = 96
    pred_len: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 50
    seed: int = 42
    early_stopping_patience: int = 10
    min_delta: float = 1e-6
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    grad_clip: float = 5.0

    # Loss hyper-parameters (see module docstring).
    mse_pretrain_epochs: int = 5
    nll_warmup_epochs: int = 10
    nll_weight_max: float = 0.1
    sharpness_alpha: float = 0.001
    var_eps: float = 1e-2

    # Architecture. These are the values behind the released results:
    # 64 conv channels / 64 RGBF hidden / 64 attention dims -> 109,250
    # trainable parameters on station01 (seq96, pred1).
    cnnrs_channels: int = 64
    rgbf_hidden: int = 64
    attn_dim: int = 64
    dropout: float = 0.1

    # Data
    nwp_features: List[str] = field(
        default_factory=lambda: [
            "ghi",
            "dhi",
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
        ]
    )
    data_root: str = "preprocessed"

    def to_dict(self) -> Dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


class UPVNetTrainer:
    def __init__(self, model: UPVNet, cfg: TrainConfig, device: str, logger=print):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.log = logger
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=cfg.scheduler_patience, factor=cfg.scheduler_factor
        )
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.history: List[Dict] = []

    def nll_weight(self, epoch: int) -> float:
        if epoch <= self.cfg.mse_pretrain_epochs:
            return 0.0
        t = epoch - self.cfg.mse_pretrain_epochs
        if self.cfg.nll_warmup_epochs <= 0:
            return float(self.cfg.nll_weight_max)
        return float(self.cfg.nll_weight_max) * min(1.0, t / float(self.cfg.nll_warmup_epochs))

    def compute_loss(self, mu: torch.Tensor, var: torch.Tensor, y: torch.Tensor, epoch: int):
        mse = F.mse_loss(mu, y)
        var = torch.clamp(var, min=self.cfg.var_eps, max=1e3)
        sigma = torch.sqrt(var)
        nll = (0.5 * torch.log(2.0 * math.pi * var) + (y - mu) ** 2 / (2.0 * var)).mean()
        sharp = sigma.mean()
        lam = self.nll_weight(epoch)
        total = mse + lam * nll + self.cfg.sharpness_alpha * sharp
        return total, {
            "loss_total": float(total.detach()),
            "loss_mse": float(mse.detach()),
            "loss_nll": float(nll.detach()),
            "loss_sharp": float(sharp.detach()),
            "lambda_nll": lam,
        }

    def run_epoch(self, loader, train: bool, epoch: int):
        self.model.train(train)
        preds, vars_, tgts = [], [], []
        total, seen, parts_acc = 0.0, 0, {}
        with torch.enable_grad() if train else torch.no_grad():
            for power_seq, nwp_seq, y in loader:
                power_seq = power_seq.to(self.device)
                nwp_seq = nwp_seq.to(self.device)
                y = y.to(self.device)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                mu, var = self.model(power_seq, nwp_seq)
                loss, parts = self.compute_loss(mu, var, y, epoch)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
                    self.optimizer.step()
                total += float(loss.detach()) * len(y)
                seen += len(y)
                for k, v in parts.items():
                    parts_acc[k] = parts_acc.get(k, 0.0) + v * len(y)
                preds.append(mu.detach().cpu().numpy())
                vars_.append(var.detach().cpu().numpy())
                tgts.append(y.detach().cpu().numpy())
        n = max(1, seen)
        return total / n, np.concatenate(preds), np.concatenate(vars_), np.concatenate(tgts), {
            k: v / n for k, v in parts_acc.items()
        }

    def fit(self, train_loader, val_loader) -> List[Dict]:
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            tr_loss, tr_pred, tr_var, tr_true, parts = self.run_epoch(train_loader, True, epoch)
            va_loss, va_pred, va_var, va_true, _ = self.run_epoch(val_loader, False, epoch)
            self.scheduler.step(va_loss)

            tr_m = compute_metrics(tr_pred, tr_true, tr_var)
            va_m = compute_metrics(va_pred, va_true, va_var)
            record = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time_sec": time.time() - t0,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "train_metrics": tr_m,
                "val_metrics": va_m,
                "loss_parts": parts,
            }
            self.history.append(record)

            improved = va_loss < self.best_val_loss - self.cfg.min_delta
            if improved:
                self.best_val_loss = va_loss
                self.best_epoch = epoch
                self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            self.log(
                f"Epoch {epoch:03d}/{self.cfg.epochs:03d} | LR={record['lr']:.2e} | "
                f"lambda={parts['lambda_nll']:.3f} | mse={parts['loss_mse']:.4f} "
                f"nll={parts['loss_nll']:.4f}"
            )
            self.log(f"  [Train] {format_metrics(tr_m)}")
            self.log(f"  [Val]   {format_metrics(va_m)}" + ("  [best]" if improved else ""))

            if epoch - self.best_epoch >= self.cfg.early_stopping_patience:
                self.log(f"[early stop] no improvement for {self.cfg.early_stopping_patience} epochs")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history


def run_experiment(cfg: TrainConfig, project_root: Path, out_dir: Path, logger=print) -> Dict:
    """Train UPVNet on one dataset/horizon and evaluate on the held-out test split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    logger(f"[config] {json.dumps(cfg.to_dict(), ensure_ascii=False)}")
    logger(f"[device] {device}")

    ddir = project_root / cfg.data_root / cfg.dataset
    splits = {}
    for name in ("train", "val", "test"):
        ds = load_csv_to_dataset(ddir / f"{name}.csv", cfg.seq_len, cfg.pred_len, cfg.nwp_features)
        if ds is None:
            raise FileNotFoundError(
                f"Could not build the {name} split from {ddir / f'{name}.csv'}. "
                "Run the preprocessing step first, and check that nwp_features match the CSV columns."
            )
        splits[name] = ds
    logger(
        f"[data] sequences train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )

    loaders = create_dataloaders(splits["train"], splits["val"], splits["test"], cfg.batch_size)

    model = UPVNet(
        nwp_dim=len(cfg.nwp_features),
        seq_len=cfg.seq_len,
        cnnrs_channels=cfg.cnnrs_channels,
        rgbf_hidden=cfg.rgbf_hidden,
        attn_dim=cfg.attn_dim,
        dropout=cfg.dropout,
    )
    param_stats = count_parameters(model)
    logger(f"[model] parameters {param_stats}")

    trainer = UPVNetTrainer(model, cfg, device, logger=logger)
    t0 = time.time()
    history = trainer.fit(loaders["train"], loaders["val"])
    train_time = time.time() - t0

    _, test_pred, test_var, test_true, _ = trainer.run_epoch(loaders["test"], False, 1)
    test_metrics = compute_metrics(test_pred, test_true, test_var)
    logger(f"[test] {format_metrics(test_metrics)}")

    summary = {
        "config": cfg.to_dict(),
        "parameters": param_stats,
        "device": device,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
        "total_train_time_sec": train_time,
        "epochs_run": len(history),
        "test_metrics": test_metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "test_predictions.npz",
        true=test_true,
        pred=test_pred,
        pred_var=test_var,
    )
    logger(f"[saved] {out_dir}")
    return summary

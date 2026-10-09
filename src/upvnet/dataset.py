"""
Dataset and data loading utilities for UPVNet.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


POWER_COL = "power"
TIMESTAMP_COL = "timestamp"


def infer_feature_cols(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> List[str]:
    if feature_cols:
        cols = [c for c in feature_cols if c in df.columns]
    else:
        cols = [c for c in df.columns if c not in {TIMESTAMP_COL, POWER_COL}]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if feature_cols and not cols:
        raise ValueError("No valid numeric NWP features found in CSV.")
    return cols


class PVSequenceDataset(Dataset):
    def __init__(self, power_seqs: np.ndarray, nwp_seqs: np.ndarray, targets: np.ndarray):
        self.power_seqs = torch.FloatTensor(power_seqs)
        self.nwp_seqs = torch.FloatTensor(nwp_seqs)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.power_seqs[idx], self.nwp_seqs[idx], self.targets[idx]


def create_sequences(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int = 1,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = df[POWER_COL].values
    feature_cols = infer_feature_cols(df, feature_cols)
    feats = df[feature_cols].values

    power_seqs: List[np.ndarray] = []
    feat_seqs: List[np.ndarray] = []
    targets: List[float] = []

    n = len(df)
    for i in range(n - seq_len - pred_len + 1):
        power_seq = power[i : i + seq_len].reshape(1, seq_len)
        feat_seq = feats[i : i + seq_len]
        target = power[i + seq_len + pred_len - 1]
        power_seqs.append(power_seq)
        feat_seqs.append(feat_seq)
        targets.append(target)

    if len(power_seqs) == 0:
        return (
            np.array([], dtype=np.float32).reshape(0, 1, seq_len),
            np.array([], dtype=np.float32).reshape(0, seq_len, len(feature_cols)),
            np.array([], dtype=np.float32),
        )

    return (
        np.array(power_seqs, dtype=np.float32),
        np.array(feat_seqs, dtype=np.float32),
        np.array(targets, dtype=np.float32),
    )


def load_csv_to_dataset(
    csv_path: Path,
    seq_len: int,
    pred_len: int,
    feature_cols: Optional[List[str]] = None,
) -> Optional[PVSequenceDataset]:
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty or len(df) < seq_len + pred_len:
        return None

    if POWER_COL not in df.columns:
        return None

    feature_cols = infer_feature_cols(df, feature_cols)
    if not feature_cols:
        return None

    power_seqs, nwp_seqs, targets = create_sequences(df, seq_len, pred_len, feature_cols)
    if len(targets) == 0:
        return None

    return PVSequenceDataset(power_seqs, nwp_seqs, targets)


def create_dataloaders(
    train_ds: PVSequenceDataset,
    val_ds: PVSequenceDataset,
    test_ds: Optional[PVSequenceDataset],
    batch_size: int,
    num_workers: int = 0,
) -> dict:
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    if test_ds is not None:
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loaders


def get_nwp_dim() -> int:
    raise NotImplementedError("NWP feature dimension is dataset-dependent; infer from CSV instead.")


def get_feature_dim_from_csv(csv_path: Path, feature_cols: Optional[List[str]] = None) -> Optional[int]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return None
    cols = infer_feature_cols(df, feature_cols)
    return len(cols) if cols else None

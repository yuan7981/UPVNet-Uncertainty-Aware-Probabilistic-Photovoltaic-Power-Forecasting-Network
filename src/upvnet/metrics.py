"""
Metrics calculation for UPVNet training.
Updated with deterministic and probabilistic metrics.
"""
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_UNITS = {
    "MAE": "(standardized)",
    "MSE": "(standardized^2)",
    "RMSE": "(standardized)",
    "MAPE": "%",
    "R2": "(dimensionless)",
    "Corr": "(dimensionless)",
    "nMAE": "%",
    "nRMSE": "%",
    "CRPS": "(standardized)",
    "PICP": "%",
    "MPIW": "(standardized)",
    "WS": "(standardized)",
    "SS": "%"
}

def compute_metrics(
    preds: np.ndarray, 
    targets: np.ndarray, 
    pred_var: Optional[np.ndarray] = None,
    ref_nrmse: Optional[float] = None  # 用于计算 SS
) -> Dict[str, Optional[float]]:
    """
    计算全面评估指标。
    :param preds: 预测均值 (mu)
    :param targets: 真实值 (y)
    :param pred_var: 预测方差 (sigma^2)，如果为 None 则不计算概率指标
    :param ref_nrmse: 参考模型的 nRMSE，用于计算 Skill Score (SS)
    """
    # 展平数组
    preds = preds.flatten()
    targets = targets.flatten()
    
    if pred_var is not None:
        pred_var = pred_var.flatten()

    if len(preds) == 0 or len(targets) == 0:
        return {k: 0.0 for k in METRIC_UNITS.keys()}

    # --- 1. 基础确定性指标 ---
    mse = float(mean_squared_error(targets, preds))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds)) if len(targets) > 1 else 0.0

    # MAPE
    non_zero_mask = np.abs(targets) > 1e-8
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((targets[non_zero_mask] - preds[non_zero_mask]) / targets[non_zero_mask])) * 100)
    else:
        mape = 0.0

    # Correlation
    if len(targets) > 1 and np.std(preds) > 1e-8 and np.std(targets) > 1e-8:
        corr, _ = pearsonr(targets, preds)
        corr = float(corr) if not np.isnan(corr) else 0.0
    else:
        corr = 0.0

    # --- 2. 归一化指标 (nMAE, nRMSE) ---
    # 公式：Metric / mean(y_true) * 100%
    target_mean = np.mean(targets)
    if abs(target_mean) > 1e-8:
        nmae = (mae / target_mean) * 100
        nrmse = (rmse / target_mean) * 100
    else:
        nmae = 0.0
        nrmse = 0.0

    # --- 3. Skill Score (SS) ---
    # SS = (1 - NRMSE_model / NRMSE_ref) * 100%
    if ref_nrmse is not None and ref_nrmse > 1e-8:
        ss = (1 - nrmse / ref_nrmse) * 100
    else:
        ss = None

    metrics = {
        "MAE": round(mae, 6),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "MAPE": round(mape, 4),
        "R2": round(r2, 6),
        "Corr": round(corr, 6),
        "nMAE": round(nmae, 4),
        "nRMSE": round(nrmse, 4),
        "SS": round(ss, 4) if ss is not None else None
    }

    # --- 4. 概率性指标 (如果提供了方差) ---
    if pred_var is not None:
        # 确保标准差有效
        std = np.sqrt(np.clip(pred_var, 1e-9, None))
        
        # 4.1 CRPS (Continuous Ranked Probability Score) - Gaussian Analytical Form
        # CRPS = sigma * [ z(2\Phi(z)-1) + 2\phi(z) - 1/sqrt(pi) ], where z = (y-mu)/sigma
        z = (targets - preds) / std
        pdf = norm.pdf(z)
        cdf = norm.cdf(z)
        crps_val = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
        crps = float(np.mean(crps_val))

        # 设定置信水平 alpha (例如 95% 置信区间)
        alpha = 0.05
        z_score = 1.96  # approx for 95%

        # 计算上下界
        lower = preds - z_score * std
        upper = preds + z_score * std

        # 4.2 PICP (Prediction Interval Coverage Probability)
        # 落在区间内的点的比例
        inside = (targets >= lower) & (targets <= upper)
        picp = float(np.mean(inside)) * 100

        # 4.3 MPIW (Mean Prediction Interval Width)
        # 平均区间宽度
        width = upper - lower
        mpiw = float(np.mean(width))

        # 4.4 WS (Winkler Score)
        # 惩罚在区间外的预测
        # L = lower, U = upper, y = target
        # if L <= y <= U: WS = U - L
        # if y < L:       WS = (U - L) + 2/alpha * (L - y)
        # if y > U:       WS = (U - L) + 2/alpha * (y - U)
        ws_scores = width.copy()
        
        mask_below = targets < lower
        mask_above = targets > upper
        
        # 惩罚项系数
        penalty_factor = 2.0 / alpha
        
        ws_scores[mask_below] += penalty_factor * (lower[mask_below] - targets[mask_below])
        ws_scores[mask_above] += penalty_factor * (targets[mask_above] - upper[mask_above])
        
        ws = float(np.mean(ws_scores))

        # 更新 Metrics 字典
        metrics.update({
            "CRPS": round(crps, 6),
            "PICP": round(picp, 4),
            "MPIW": round(mpiw, 6),
            "WS": round(ws, 6)
        })
    else:
        # 如果没有方差，这些指标为空
        metrics.update({
            "CRPS": None, "PICP": None, "MPIW": None, "WS": None
        })

    return metrics

def format_metrics(metrics: Dict[str, Optional[float]], prefix: str = "") -> str:
    parts = []
    # 定义打印顺序，常用的在前
    order = ["MAE", "RMSE", "R2", "nRMSE", "CRPS", "PICP", "WS"]
    
    # 先打印重点指标
    for k in order:
        if k in metrics and metrics[k] is not None:
            parts.append(f"{k}={metrics[k]:.4f}")
    
    # 打印剩余指标
    for k, v in metrics.items():
        if k not in order and v is not None:
             parts.append(f"{k}={v:.4f}")
             
    return f"{prefix}{' | '.join(parts)}"

def verify_metrics(preds: np.ndarray, targets: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """保持原有的验证逻辑，简单调用 compute_metrics 即可"""
    # 这里为了兼容旧代码接口，我们暂时只返回确定性指标
    return compute_metrics(preds, targets, pred_var=None)
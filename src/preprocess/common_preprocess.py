import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import ParserError
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ================= 核心配置 =================
COMMON_COLUMNS = [
    "timestamp",
    "power",
    "ghi",
    "dhi",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "season_cluster",
    "weather_cluster"
]

SEASON_MAP = {
    0: "Spring",
    1: "Summer",
    2: "Fall",
    3: "Winter"
}

@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    full: pd.DataFrame
    scaler: StandardScaler
    summary: dict
    missing_summary: pd.DataFrame

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _safe_float(v):
    if hasattr(v, "item"):
        return v.item()
    return v

def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_safe_float)

def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")

# 保持原有的构建函数不变
def build_common_from_pvod(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["date_time"], errors="coerce")
    out["power"] = df.get("power")
    out["ghi"] = df.get("lmd_totalirrad")
    out.loc[out["ghi"].isna(), "ghi"] = df.get("nwp_globalirrad")
    out["dhi"] = df.get("lmd_diffuseirrad")
    out["temperature"] = df.get("lmd_temperature")
    out.loc[out["temperature"].isna(), "temperature"] = df.get("nwp_temperature")
    out["humidity"] = df.get("nwp_humidity")
    out["wind_speed"] = df.get("lmd_windspeed")
    out.loc[out["wind_speed"].isna(), "wind_speed"] = df.get("nwp_windspeed")
    out["wind_direction"] = df.get("lmd_winddirection")
    out.loc[out["wind_direction"].isna(), "wind_direction"] = df.get("nwp_winddirection")
    return out

def build_common_from_dka(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    out["power"] = df.get("Active_Power")
    out["ghi"] = df.get("Global_Horizontal_Radiation")
    out["dhi"] = df.get("Diffuse_Horizontal_Radiation")
    out["temperature"] = df.get("Weather_Temperature_Celsius")
    out["humidity"] = df.get("Weather_Relative_Humidity")
    out["wind_speed"] = df.get("Wind_Speed")
    out["wind_direction"] = df.get("Wind_Direction")
    return out

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return df

def get_season(month: int, hemisphere: Literal['north', 'south'] = 'north') -> int:
    """
    根据半球返回季节索引
    0:Spring, 1:Summer, 2:Fall, 3:Winter
    """
    if hemisphere == 'north':
        # 北半球: 春(3-5), 夏(6-8), 秋(9-11), 冬(12,1,2)
        if 3 <= month <= 5: return 0
        elif 6 <= month <= 8: return 1
        elif 9 <= month <= 11: return 2
        else: return 3
    else:
        # 南半球 (如澳大利亚): 春(9-11), 夏(12,1,2), 秋(3-5), 冬(6-8)
        if 9 <= month <= 11: return 0  # Spring
        elif month == 12 or month <= 2: return 1  # Summer
        elif 3 <= month <= 5: return 2  # Fall
        else: return 3  # Winter

def perform_clustering(df: pd.DataFrame, hemisphere: str = 'north') -> pd.DataFrame:
    df = df.copy()
    
    # 1. 季节聚类 (传入半球参数)
    df["season_cluster"] = df.index.month.map(lambda x: get_season(x, hemisphere))
    
    # 2. 天气聚类 (仅基于物理变量)
    cluster_features = ["ghi", "temperature", "humidity"]
    available_feats = [c for c in cluster_features if c in df.columns]
    
    if len(available_feats) >= 2:
        temp_data = df[available_feats].fillna(0)
        scaler_temp = StandardScaler()
        X_cluster = scaler_temp.fit_transform(temp_data)
        
        # 聚成4类 (e.g. 晴/多云/阴/雨)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df["weather_cluster"] = kmeans.fit_predict(X_cluster)
    else:
        df["weather_cluster"] = 0
        
    return df

def preprocess_and_split(
    df_raw: pd.DataFrame, 
    resample_rule: Optional[str], 
    downsample_from: Optional[str] = None,
    hemisphere: str = 'north'  # 新增参数
) -> SplitData:
    
    # 1. 基础清洗与重采样
    df = df_raw.copy()
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.duplicated(subset=["timestamp"]).sum() > 0:
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.set_index("timestamp").sort_index()

    if downsample_from and resample_rule:
        df = df.resample(resample_rule).asfreq()
        print(f"[INFO] Downsampled from {downsample_from} to {resample_rule}")
    elif resample_rule:
        df = df.resample(resample_rule).asfreq()

    # 2. 物理约束与清洗
    # [改进1] 夜间噪声过滤：当 GHI < 10 (即天黑) 时，强制 Power = 0
    if "ghi" in df.columns and "power" in df.columns:
        # 注意：这里假设 GHI 单位是 W/m2。如果数据里是 kW/m2，阈值需改为 0.01
        mask_night = df["ghi"] < 10.0
        df.loc[mask_night, "power"] = 0.0
        
    for col in ["power", "ghi", "dhi", "wind_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    for col in ["temperature", "humidity", "wind_direction"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. 插补 (Interpolation)
    # [改进2] 增加 limit 限制，防止长时缺失被错误地画成直线
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # limit=4 (假设15min数据，即1小时)。超过1小时的缺失保留NaN，避免制造伪数据
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=4, limit_direction="both")
    
    # 统计仍缺失的数据（超过limit限制的部分）
    missing_summary = (
        df.isna().sum().rename("null_count").to_frame()
        .assign(null_rate=lambda x: x["null_count"] / len(df))
        .reset_index().rename(columns={"index": "column"})
        .sort_values("null_rate", ascending=False)
    )

    # 4. 特征与聚类
    df = add_time_features(df)
    # [改进3] 传入半球信息，确保季节标签正确
    df = perform_clustering(df, hemisphere=hemisphere)
    
    # 补齐缺少的列
    for c in COMMON_COLUMNS:
        if c not in df.columns and c != "timestamp":
            df[c] = 0.0
            
    cleaned = df.reset_index()[COMMON_COLUMNS]

    # 5. 切分 (Train/Val/Test) — 7:1:2
    n = len(cleaned)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    train_df = cleaned.iloc[:train_end].copy()
    val_df = cleaned.iloc[train_end:val_end].copy()
    test_df = cleaned.iloc[val_end:].copy()
    
    # 6. 数据泄露防护标准化 (Fit on Train ONLY)
    exclude_cols = ["timestamp", "season_cluster", "weather_cluster"]
    scale_cols = [c for c in COMMON_COLUMNS if c not in exclude_cols]
    
    scaler = StandardScaler()
    
    # 仅在有数据的 Train 上拟合
    # 注意：由于interpolate加了limit，这里可能仍有NaN，StandardScaler不能处理NaN
    # 策略：Fit时先fillna(0)，这比crash好，也是工业界常用做法
    if len(train_df) > 0:
        scaler.fit(train_df[scale_cols].fillna(0))
    
    # Transform 所有数据集
    for sub_df in [train_df, val_df, test_df, cleaned]:
        if len(sub_df) > 0:
            sub_df[scale_cols] = scaler.transform(sub_df[scale_cols].fillna(0))
            
    summary = {
        "n_rows": int(n),
        "start": str(cleaned["timestamp"].min()) if len(cleaned) else None,
        "end": str(cleaned["timestamp"].max()) if len(cleaned) else None,
        "scaler_mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else [],
        "scaler_scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
        "hemisphere_used": hemisphere
    }

    return SplitData(train_df, val_df, test_df, cleaned, scaler, summary, missing_summary)

# ================= 可视化模块 (保持增强版) =================

def plot_visualizations(df: pd.DataFrame, output_dir: str, prefix: str = ""):
    if len(df) == 0: return
    figures_dir = os.path.join(output_dir, "figures")
    _ensure_dir(figures_dir)
    
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 1. 散点图: GHI vs Power
    if "ghi" in df.columns and "power" in df.columns:
        plt.figure(figsize=(6, 6))
        sample = df.sample(min(2000, len(df)), random_state=42)
        plt.scatter(sample["ghi"], sample["power"], alpha=0.3, s=10)
        plt.xlabel("GHI (Std)")
        plt.ylabel("Power (Std)")
        plt.title(f"{prefix}GHI vs Power")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{prefix}scatter_ghi_power.png"))
        plt.close()

    # 2. 热力图
    if "power" in df.columns:
        try:
            temp = df.copy()
            temp["date"] = temp["timestamp"].dt.date
            temp["hour"] = temp["timestamp"].dt.hour
            pivot = temp.pivot_table(index="hour", columns="date", values="power", aggfunc="mean")
            plt.figure(figsize=(10, 5))
            sns.heatmap(pivot, cmap="inferno", cbar=False, xticklabels=False)
            plt.title(f"{prefix}Power Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"{prefix}heatmap_power.png"))
            plt.close()
        except: pass

    # 3. 箱线图
    if "power" in df.columns:
        plt.figure(figsize=(8, 4))
        temp = df.copy()
        temp["hour"] = temp["timestamp"].dt.hour
        sns.boxplot(data=temp, x="hour", y="power", showfliers=False)
        plt.title(f"{prefix}Hourly Power Boxplot")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{prefix}boxplot_hourly.png"))
        plt.close()

def plot_clustering_overview(df: pd.DataFrame, output_dir: str):
    if len(df) == 0: return
    cluster_figs_dir = os.path.join(output_dir, "clustering_analysis")
    _ensure_dir(cluster_figs_dir)
    
    sample = df.sample(min(5000, len(df)), random_state=42)
    
    # 聚类散点图
    if "ghi" in df.columns and "humidity" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=sample, x="ghi", y="humidity", hue="weather_cluster", palette="viridis", s=15, alpha=0.6)
        plt.title("Weather Clustering: GHI vs Humidity")
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_figs_dir, "weather_cluster_scatter.png"))
        plt.close()

    # 季节箱线图
    if "power" in df.columns and "season_cluster" in df.columns:
        plt.figure(figsize=(8, 5))
        temp = sample.copy()
        temp["season_name"] = temp["season_cluster"].map(SEASON_MAP)
        order = ["Spring", "Summer", "Fall", "Winter"]
        sns.boxplot(data=temp, x="season_name", y="power", order=order, palette="Set2")
        plt.title("Power Distribution by Season")
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_figs_dir, "season_power_boxplot.png"))
        plt.close()

def save_and_analyze_subsets(df: pd.DataFrame, root_dir: str, group_col: str, folder_name: str, name_map: dict = None):
    base_path = os.path.join(root_dir, folder_name)
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    _ensure_dir(base_path)
    
    groups = df.groupby(group_col)
    for gid, sub_df in groups:
        if name_map:
            name_suffix = name_map.get(int(gid), str(gid))
        else:
            name_suffix = str(int(gid))
            
        sub_dir_name = f"{folder_name}_{name_suffix}"
        sub_dir_path = os.path.join(base_path, sub_dir_name)
        _ensure_dir(sub_dir_path)
        
        csv_name = f"{sub_dir_name}.csv"
        sub_df.to_csv(os.path.join(sub_dir_path, csv_name), index=False)
        plot_visualizations(sub_df, sub_dir_path, prefix=f"[{name_suffix}] ")
        
    print(f"[INFO] Saved & Visualized {len(groups)} subsets in {base_path}")

def save_artifacts(data: SplitData, output_dir: str, dataset_name: str) -> None:
    _ensure_dir(output_dir)
    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)
    
    # 保存基础切分
    data.train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    data.val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    data.test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # 保存元数据
    data.missing_summary.to_csv(os.path.join(tables_dir, "missing_summary.csv"), index=False)
    _write_json(os.path.join(output_dir, "summary.json"), data.summary)

    # 基础可视化
    plot_visualizations(data.train, output_dir, prefix="train_")
    
    # 聚类总览
    plot_clustering_overview(data.full, output_dir)
    
    # 分类保存 Season
    save_and_analyze_subsets(
        df=data.full, 
        root_dir=output_dir, 
        group_col="season_cluster", 
        folder_name="season_data",
        name_map=SEASON_MAP
    )
    
    # 分类保存 Weather
    save_and_analyze_subsets(
        df=data.full, 
        root_dir=output_dir, 
        group_col="weather_cluster", 
        folder_name="weather_data",
        name_map=None
    )

    print(f"[INFO] {dataset_name} processing complete. (Hemisphere: {data.summary['hemisphere_used']})")
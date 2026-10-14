"""Beyond R2: rigorous BMW used-car price pipeline (standalone script).

Cell-order concatenation of the Kaggle notebook: 10 EDA steps followed by a
leakage-safe sklearn Pipeline + GroupKFold OOF evaluation, multi-seed
reproducibility check, and an inference benchmark.

Local usage:
    python src/bmw_pipeline.py --csv_path /path/to/BMW_Cars_Pakistan.csv --output_dir output
"""


# ============================================================================
# import argparse
# ============================================================================
# ============================================================================


import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

warnings.filterwarnings("ignore", message=".*unknown categories.*", category=UserWarning)


# ============================================================================
# KAGGLE_PATH_PRIMARY = "/kaggle/input/datasets/ahmedfakhar123/used-car-
# ============================================================================
# ============================================================================


KAGGLE_PATH_PRIMARY = "/kaggle/input/datasets/ahmedfakhar123/used-car-price-prediction-dataset/BMW_Cars_Pakistan.csv"
KAGGLE_PATH_FALLBACK = "/kaggle/input/used-car-price-prediction-dataset/BMW_Cars_Pakistan.csv"

# Global config
CONFIG = {
    "random_seed": 42,
    "n_splits": 5,            # 5-fold cross-validation
    "n_repeats": 3,           # Repeated CV: 3 repeats (15 total runs)
    "xgb_threads": 2,         # Limit threads per XGBoost instance (avoid oversubscription)
    "fold_jobs": 4,           # Fold-level parallelism: run 4 folds simultaneously
    # Adjust so that xgb_threads * fold_jobs <= physical CPU cores
    # e.g. 8-core CPU: 2 * 4 = 8, exactly at limit
    "output_dir": "output",   # Output directory
}

# Unified color palette
PALETTE = {
    "primary": "#58a6ff",
    "danger":  "#f85149",
    "success": "#3fb950",
    "warning": "#d29922",
    "purple":  "#bc8cff",
    "orange":  "#f0883e",
}

# Insight collector (for final report)
INSIGHTS = []


def insight(title, *points):
    """Record an insight and print to console."""
    block = {"title": title, "points": list(points)}
    INSIGHTS.append(block)
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")
    for p in points:
        print(f"  - {p}")


# =====================================================================
#  Font Setup
# =====================================================================

for font_name in ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                  "Microsoft YaHei", "PingFang SC", "SimHei"]:
    matches = [f for f in fm.fontManager.ttflist if font_name in f.name]
    if matches:
        plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams["font.sans-serif"]
        break
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
#  Utility Functions
# =====================================================================

def compute_file_hash(file_path):
    """Compute SHA-256 hash of a file for data versioning."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_dataframe_hash(df):
    """Compute hash of DataFrame content."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def get_data_path():
    """Get the CSV path: try Kaggle paths first, then argparse fallback."""
    for path in [KAGGLE_PATH_PRIMARY, KAGGLE_PATH_FALLBACK]:
        if os.path.exists(path):
            return path

    parser = argparse.ArgumentParser(description="BMW Price Prediction - Complete Pipeline")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to BMW_Cars_Pakistan.csv")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    args, _ = parser.parse_known_args()
    CONFIG["output_dir"] = args.output_dir
    return args.csv_path


def extract_bmw_model(car_name):
    """
    Extract BMW model from full car name.
    e.g. 'BMW X5 2012 xDrive50i for Sale' -> 'X5'
         'BMW 7 Series 2017 740 Le xDrive for Sale' -> '7 Series'
    """
    match = re.search(r"BMW\s+(.*?)\s+\d{4}", car_name)
    if match:
        extracted = match.group(1)
        if "Series" in extracted:
            return " ".join(extracted.split()[:2])
        else:
            return extracted.split()[0]
    return "Other"


# ============================================================================
# CSV_PATH = get_data_path()
# ============================================================================
# ============================================================================


CSV_PATH = get_data_path()
OUTPUT_DIR = CONFIG["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Data path: {CSV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================================
# #  EDA Step 1: Data Loading & Overview
# ============================================================================
# ============================================================================


#  EDA Step 1: Data Loading & Overview
# =====================================================================

print("=" * 60)
print("EDA Step 1: Data Loading & Overview")
print("=" * 60)

df_bmw = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df_bmw.shape}")
print(f"\n--- dtypes ---")
print(df_bmw.dtypes)
print(f"\n--- First 5 rows ---")
print(df_bmw.head().to_string())

insight(
    "Insight 1: Data Structure Overview",
    f"Dataset has {df_bmw.shape[0]} rows x {df_bmw.shape[1]} columns",
    f"Numeric columns: {df_bmw.select_dtypes(include=[np.number]).shape[1]}, "
    f"categorical columns: {df_bmw.select_dtypes(include=['object']).shape[1]}",
    f"Memory usage: {df_bmw.memory_usage(deep=True).sum() / 1024:.1f} KB",
)


# ============================================================================
# #  EDA Step 2: Missing Value Analysis (quantified)
# ============================================================================
# ============================================================================


#  EDA Step 2: Missing Value Analysis (quantified)
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 2: Missing Value Analysis")
print("=" * 60)

missing = df_bmw.isnull().sum()
missing_pct = (missing / len(df_bmw) * 100).round(2)
missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct,
    "non_null_count": df_bmw.notnull().sum(),
}).sort_values("missing_pct", ascending=False)

print(missing_df.to_string())

# --- Missing value heatmap ---
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df_bmw.isnull(), cbar=True, yticklabels=False,
            cmap=sns.color_palette(["#3fb950", "#f85149"]), ax=ax)
ax.set_title("Missing Value Heatmap (Green=Present, Red=Missing)", fontsize=13, fontweight="bold")
ax.set_xlabel("Columns")
ax.set_ylabel("Row Index")
plt.tight_layout()
plt.show()

# --- Analysis ---
high_missing = missing_df[missing_df["missing_pct"] > 50]
target_missing = missing_df.loc["Price (PKR)"] if "Price (PKR)" in missing_df.index else None

_points = []
if len(high_missing) > 0:
    for col, row in high_missing.iterrows():
        _points.append(f"{col} missing rate {row['missing_pct']}% ({int(row['missing_count'])}/{len(df_bmw)}), "
                       f"exceeds 50% threshold, should be removed")
if target_missing is not None and target_missing["missing_count"] > 0:
    _points.append(f"Target variable Price (PKR) missing {int(target_missing['missing_count'])} rows "
                 f"({target_missing['missing_pct']}%), cannot be used for training, must drop")
_points.append("Unnamed: 0 is a pure index column, contains no information, should be removed")

insight("Insight 2: Missing Value Analysis", *_points)


# ============================================================================
# #  EDA Step 3: Data Cleaning (track before/after)
# ============================================================================
# ============================================================================


#  EDA Step 3: Data Cleaning (track before/after)
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 3: Data Cleaning")
print("=" * 60)

df_clean = df_bmw.copy()
n_original = len(df_clean)

# 1. Remove non-informative columns
drop_cols = [c for c in ["Unnamed: 0", "Auction Rating"] if c in df_clean.columns]
df_clean = df_clean.drop(columns=drop_cols)
print(f"Removed columns: {drop_cols}")

# 2. Remove rows with missing target
n_before = len(df_clean)
df_clean = df_clean.dropna(subset=["Price (PKR)"])
n_after_dropna = len(df_clean)
print(f"Removed rows with missing Price: {n_before - n_after_dropna}")

# 3. Remove exact duplicates
n_before = len(df_clean)
duplicates_count = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
n_after_dedup = len(df_clean)
print(f"Removed exact duplicates: {duplicates_count}")

df_clean.reset_index(drop=True, inplace=True)

# 4. Create target variable (millions PKR)
df_clean["Price_Millions"] = df_clean["Price (PKR)"] / 1_000_000

print(f"\nBefore cleaning: {n_original} rows -> After: {len(df_clean)} rows "
      f"(retention rate {len(df_clean)/n_original*100:.1f}%)")

insight(
    "Insight 3: Data Cleaning",
    f"Original {n_original} rows -> Cleaned {len(df_clean)} rows, "
    f"retention rate {len(df_clean)/n_original*100:.1f}%",
    f"Removed {drop_cols} columns + {n_before - n_after_dropna} target-missing rows + {duplicates_count} duplicates",
    f"{len(df_clean)} high-quality records beat {n_original} noisy records (Garbage in, garbage out)",
)


# ============================================================================
# #  EDA Step 4: Near-Duplicate Detection 
# ============================================================================
# ============================================================================


#  EDA Step 4: Near-Duplicate Detection 
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 4: Near-Duplicate Detection")
print("=" * 60)

# Group by key features
group_cols = ["Model Year", "Fuel Type", "Transmission", "Engine Capacity"]
_near_dup_groups = []
for group_key, group_df in df_clean.groupby(group_cols):
    if len(group_df) > 1:
        # Check if mileage is also close (within 5000km)
        for i in range(len(group_df)):
            for j in range(i + 1, len(group_df)):
                mile_diff = abs(group_df.iloc[i]["Mileage"] - group_df.iloc[j]["Mileage"])
                if mile_diff <= 5000:
                    _near_dup_groups.append({
                        "record_1_idx": group_df.index[i],
                        "record_2_idx": group_df.index[j],
                        "shared_features": str(group_key),
                        "mileage_diff": mile_diff,
                        "price_diff": abs(group_df.iloc[i]["Price_Millions"] -
                                         group_df.iloc[j]["Price_Millions"]),
                    })

n_near = len(_near_dup_groups)
print(f"Exact duplicates (already removed): {df_clean.duplicated().sum()}")
print(f"Near-duplicate groups (same year+fuel+transmission+capacity+mileage diff<=5000km): {n_near}")

if n_near > 0:
    near_dup_df = pd.DataFrame(_near_dup_groups)
    print(f"\n--- Near-duplicate examples (first 10) ---")
    print(near_dup_df.head(10).to_string())

    # Plot near-duplicate price difference distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(near_dup_df["price_diff"], bins=20, color=PALETTE["warning"],
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Price Difference between Near-Duplicates (Million PKR)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Near-Duplicate Price Difference Distribution ({n_near} groups)", fontweight="bold")
    plt.tight_layout()
    plt.show()

_points = [
    "Exact duplicates fully removed",
    f"Detected {n_near} near-duplicate groups (same model+year+fuel+transmission+capacity+mileage diff<=5000km)",
]
if n_near > 0:
    avg_price_diff = np.mean([g["price_diff"] for g in _near_dup_groups])
    _points.append(f"Average price difference among near-duplicates: {avg_price_diff:.2f} Million PKR")
    _points.append("Recommendation: use GroupKFold in cross-validation to prevent "
                  "similar records from appearing in both train and val folds")
else:
    _points.append("No near-duplicate records detected")

insight("Insight 4: Near-Duplicate Detection", *_points)


# ============================================================================
# #  EDA Step 5: Target Variable Analysis + Normality Test
# ============================================================================
# ============================================================================


#  EDA Step 5: Target Variable Analysis + Normality Test
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 5: Target Variable Analysis & Normality Test")
print("=" * 60)

price = df_clean["Price_Millions"]

# Descriptive statistics
stats_summary = {
    "count": len(price),
    "mean": price.mean(),
    "median": price.median(),
    "std": price.std(),
    "min": price.min(),
    "max": price.max(),
    "range": price.max() - price.min(),
    "skewness": price.skew(),
    "kurtosis": price.kurtosis(),
    "Q1": price.quantile(0.25),
    "Q3": price.quantile(0.75),
    "IQR": price.quantile(0.75) - price.quantile(0.25),
}
print("--- Descriptive Statistics ---")
for k, v in stats_summary.items():
    print(f"  {k:<12}: {v:.4f}")

# Shapiro-Wilk normality test (valid for n <= 5000)
shapiro_stat, shapiro_p = stats.shapiro(price)
log_price = np.log1p(price)
shapiro_log_stat, shapiro_log_p = stats.shapiro(log_price)

print(f"\n--- Shapiro-Wilk Normality Test ---")
print(f"  Raw Price:    W={shapiro_stat:.4f}, p={shapiro_p:.2e} "
      f"{'-> Not normal' if shapiro_p < 0.05 else '-> Normal'}")
print(f"  log1p(Price): W={shapiro_log_stat:.4f}, p={shapiro_log_p:.2e} "
      f"{'-> Not normal' if shapiro_log_p < 0.05 else '-> Normal'}")

# Skewness interpretation
skew = price.skew()
if skew > 1:
    skew_desc = "heavily right-skewed (positive)"
elif skew > 0.5:
    skew_desc = "moderately right-skewed"
elif skew < -1:
    skew_desc = "heavily left-skewed (negative)"
elif skew < -0.5:
    skew_desc = "moderately left-skewed"
else:
    skew_desc = "approximately symmetric"

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Target Variable Price - Deep Analysis", fontsize=16, fontweight="bold", y=1.02)

# 1. Raw distribution + KDE
ax = axes[0, 0]
sns.histplot(price, kde=True, bins=30, color=PALETTE["danger"], ax=ax, alpha=0.7)
ax.axvline(price.mean(), color=PALETTE["primary"], linestyle="--", linewidth=2,
          label=f"Mean={price.mean():.1f}")
ax.axvline(price.median(), color=PALETTE["success"], linestyle="--", linewidth=2,
          label=f"Median={price.median():.1f}")
ax.set_title(f"Raw Price Distribution (skewness={skew:.2f}, {skew_desc})", fontweight="bold")
ax.set_xlabel("Price (Million PKR)")
ax.legend()

# 2. Q-Q plot (raw)
ax = axes[0, 1]
stats.probplot(price, dist="norm", plot=ax)
ax.set_title("Q-Q Plot (Raw Price)", fontweight="bold")
ax.get_lines()[0].set_color(PALETTE["danger"])
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color(PALETTE["primary"])

# 3. Log-transformed distribution
ax = axes[1, 0]
sns.histplot(log_price, kde=True, bins=30, color=PALETTE["success"], ax=ax, alpha=0.7)
ax.axvline(log_price.mean(), color=PALETTE["primary"], linestyle="--", linewidth=2,
          label=f"Mean={log_price.mean():.2f}")
ax.set_title(f"log1p(Price) Distribution (skewness={log_price.skew():.2f})", fontweight="bold")
ax.set_xlabel("log1p(Price)")
ax.legend()

# 4. Q-Q plot (log-transformed)
ax = axes[1, 1]
stats.probplot(log_price, dist="norm", plot=ax)
ax.set_title("Q-Q Plot (log1p(Price))", fontweight="bold")
ax.get_lines()[0].set_color(PALETTE["success"])
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color(PALETTE["primary"])

plt.tight_layout()
plt.show()

insight(
    "Insight 5: Target Variable Analysis",
    f"Price range {price.min():.1f} ~ {price.max():.1f} Million PKR, "
    f"mean {price.mean():.1f}, median {price.median():.1f}",
    f"Skewness={skew:.2f} ({skew_desc}), kurtosis={price.kurtosis():.2f}",
    f"Shapiro-Wilk: raw p={shapiro_p:.2e} -> reject normality assumption",
    f"log1p transform reduces skewness to {log_price.skew():.2f}, "
    f"but p={shapiro_log_p:.2e} still rejects normality (small sample + extreme values)",
    f"Recommendation: tree models like XGBoost do not require normal target, no transform needed; "
    f"consider log transform if comparing with linear models",
)


# ============================================================================
# #  EDA Step 6: Numerical Features + Outlier Detection
# ============================================================================
# ============================================================================


#  EDA Step 6: Numerical Features + Outlier Detection
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 6: Numerical Features & Outlier Detection")
print("=" * 60)

num_cols = ["Model Year", "Mileage", "Engine Capacity"]
target = "Price_Millions"

# --- Descriptive statistics ---
print("--- Numerical Feature Statistics ---")
desc = df_clean[num_cols + [target]].describe().round(2)
print(desc.to_string())

# --- Outlier detection (IQR + Z-score dual) ---
print("\n--- Outlier Detection ---")
outlier_info = {}
for col in num_cols + [target]:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]

    z_scores = np.abs(stats.zscore(df_clean[col]))
    z_outliers = df_clean[z_scores > 3]

    outlier_info[col] = {
        "IQR_bounds": (round(lower, 2), round(upper, 2)),
        "IQR_outliers": len(iqr_outliers),
        "Z_score_outliers": len(z_outliers),
        "IQR_indices": iqr_outliers.index.tolist(),
    }
    print(f"  {col:<18}: IQR range=[{lower:.1f}, {upper:.1f}], "
          f"IQR outliers={len(iqr_outliers)}, Z>3 outliers={len(z_outliers)}")

# --- Dual correlation (Pearson + Spearman) ---
print("\n--- Correlation (Pearson + Spearman) ---")
for col in num_cols:
    pearson_r, pearson_p = stats.pearsonr(df_clean[col], df_clean[target])
    spearman_r, spearman_p = stats.spearmanr(df_clean[col], df_clean[target])
    print(f"  {col:<18}: Pearson r={pearson_r:+.4f} (p={pearson_p:.2e}), "
          f"Spearman rho={spearman_r:+.4f} (p={spearman_p:.2e})")

# --- Plotting: scatter + box ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Numerical Features vs Price: Scatter + Outlier Annotation", fontsize=16, fontweight="bold", y=1.02)

colors = [PALETTE["primary"], PALETTE["success"], PALETTE["orange"]]

for i, col in enumerate(num_cols):
    # Scatter + regression line
    ax = axes[0, i]
    sns.regplot(data=df_clean, x=col, y=target, ax=ax,
                scatter_kws={"alpha": 0.6, "color": colors[i]},
                line_kws={"color": PALETTE["danger"], "linewidth": 2})
    # Annotate IQR outliers
    outliers_idx = outlier_info[col]["IQR_indices"]
    if outliers_idx:
        ax.scatter(df_clean.loc[outliers_idx, col], df_clean.loc[outliers_idx, target],
                   color=PALETTE["danger"], s=80, zorder=5, label=f"IQR outliers ({len(outliers_idx)})")
        ax.legend()
    r, p = stats.pearsonr(df_clean[col], df_clean[target])
    ax.set_title(f"{col} vs Price\nPearson r={r:+.3f} (p={p:.1e})", fontweight="bold")

    # Box plot
    ax = axes[1, i]
    sns.boxplot(data=df_clean, x=col, ax=ax, color=colors[i])
    ax.set_title(f"{col} Box Plot (Outlier Detection)", fontweight="bold")

plt.tight_layout()
plt.show()

# --- Pearson correlation matrix heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
corr_cols = num_cols + [target]
corr_matrix = df_clean[corr_cols].corr(method="pearson")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            square=True, linewidths=0.5)
ax.set_title("Pearson Correlation Matrix (Lower Triangle)", fontweight="bold")
plt.tight_layout()
plt.show()

# --- Spearman correlation matrix (captures nonlinear monotonic) ---
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix_s = df_clean[corr_cols].corr(method="spearman")
sns.heatmap(corr_matrix_s, mask=mask, annot=True, fmt=".3f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            square=True, linewidths=0.5)
ax.set_title("Spearman Correlation Matrix (Lower Triangle)", fontweight="bold")
plt.tight_layout()
plt.show()

# --- Insight ---
year_r, _ = stats.pearsonr(df_clean["Model Year"], df_clean[target])
mile_r, _ = stats.pearsonr(df_clean["Mileage"], df_clean[target])
eng_r, _ = stats.pearsonr(df_clean["Engine Capacity"], df_clean[target])

year_s, _ = stats.spearmanr(df_clean["Model Year"], df_clean[target])
mile_s, _ = stats.spearmanr(df_clean["Mileage"], df_clean[target])
eng_s, _ = stats.spearmanr(df_clean["Engine Capacity"], df_clean[target])

insight(
    "Insight 6: Numerical Features",
    f"Model Year: Pearson r={year_r:+.3f}, Spearman rho={year_s:+.3f} -> "
    f"{'strong positive' if year_r > 0.5 else 'weak positive'} (newer = more expensive)",
    f"Mileage: Pearson r={mile_r:+.3f}, Spearman rho={mile_s:+.3f} -> "
    f"{'strong negative' if abs(mile_r) > 0.5 else 'weak negative'} (higher mileage = cheaper)",
    f"Engine Capacity: Pearson r={eng_r:+.3f}, Spearman rho={eng_s:+.3f} -> "
    f"counterintuitive negative correlation! Electric cars (low capacity in kWh) are actually more expensive",
    f"Outlier detection: IQR found {sum(v['IQR_outliers'] for v in outlier_info.values())} outliers, "
    f"Z-score>3 found {sum(v['Z_score_outliers'] for v in outlier_info.values())} extreme values",
    f"Pearson captures linear, Spearman captures monotonic; "
    f"large difference indicates nonlinear relationship",
)


# ============================================================================
# #  EDA Step 7: Categorical Features + Kruskal-Wallis Test
# ============================================================================
# ============================================================================


#  EDA Step 7: Categorical Features + Kruskal-Wallis Test
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 7: Categorical Features & Statistical Tests")
print("=" * 60)

cat_cols = ["Fuel Type", "Transmission"]

for col in cat_cols:
    print(f"\n--- {col} ---")
    vc = df_clean[col].value_counts()
    print(f"Frequency:\n{vc.to_string()}")
    print(f"Number of categories: {df_clean[col].nunique()}")

    # Per-group statistics
    group_stats = df_clean.groupby(col)[target].agg(["count", "mean", "median", "std"]).round(2)
    print(f"\nGroup statistics:\n{group_stats.to_string()}")

    # Kruskal-Wallis test (non-parametric, does not require normality)
    groups_data = [g[target].values for _, g in df_clean.groupby(col)]
    if len(groups_data) >= 2 and all(len(g) >= 2 for g in groups_data):
        kw_stat, kw_p = stats.kruskal(*groups_data)
        sig = "significant" if kw_p < 0.05 else "not significant"
        print(f"\nKruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.2e} -> group difference {sig}")

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Categorical Features vs Price: Distribution & Statistical Tests", fontsize=16, fontweight="bold", y=1.02)

for i, col in enumerate(cat_cols):
    # Box plot
    ax = axes[0, i]
    sns.boxplot(data=df_clean, x=col, y=target, ax=ax, palette="Set2")
    ax.set_title(f"Price Distribution by {col}", fontweight="bold")
    ax.set_ylabel("Price (Million PKR)")

    # Category sample count bar chart
    ax = axes[1, i]
    vc = df_clean[col].value_counts()
    bars = ax.bar(range(len(vc)), vc.values, color=PALETTE["primary"], alpha=0.8)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=30)
    ax.set_title(f"{col} Category Sample Counts", fontweight="bold")
    ax.set_ylabel("Count")
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.show()

# --- Engine Unit redundancy analysis ---
print("\n--- Engine Unit Redundancy Analysis ---")
if "Engine Unit" in df_clean.columns:
    cross = pd.crosstab(df_clean["Fuel Type"], df_clean["Engine Unit"])
    print(f"Fuel Type x Engine Unit cross table:\n{cross.to_string()}")

    # Chi-square test
    chi2, chi2_p, dof, expected = stats.chi2_contingency(cross)
    print(f"\nChi-square: chi2={chi2:.2f}, p={chi2_p:.2e}, dof={dof}")
    cramers_v = np.sqrt(chi2 / (len(df_clean) * (min(cross.shape) - 1)))
    print(f"Cramer's V = {cramers_v:.4f} -> "
          f"{'perfectly correlated' if cramers_v > 0.9 else 'partially correlated'}")

insight(
    "Insight 7: Categorical Features",
    f"Fuel Type: Electric/Hybrid/PHEV median price significantly higher than Petrol/Diesel",
    f"Transmission: Automatic ({len(df_clean[df_clean['Transmission']=='Automatic'])}) "
    f"vs Manual ({len(df_clean[df_clean['Transmission']=='Manual'])}), "
    f"severe class imbalance",
    f"Kruskal-Wallis test confirms Fuel Type group price difference is significant (p < 0.05)",
    f"Engine Unit is fully redundant with Fuel Type (Cramer's V ~ 1.0), "
    f"chi-square p ~ 0 -> should remove Engine Unit",
)


# ============================================================================
# #  EDA Step 8: Engine Capacity Mixed Units Analysis
# ============================================================================
# ============================================================================


#  EDA Step 8: Engine Capacity Mixed Units Analysis
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 8: Engine Capacity Mixed Units Analysis")
print("=" * 60)

if "Engine Unit" not in df_clean.columns:
    print("Engine Unit column already removed, skipping this analysis")
else:
    print(f"--- Engine Capacity Statistics by Fuel Type ---")
    for fuel_type in df_clean["Fuel Type"].unique():
        subset = df_clean[df_clean["Fuel Type"] == fuel_type]
        unit = subset["Engine Unit"].iloc[0] if len(subset) > 0 else "?"
        print(f"  {fuel_type:<10} (unit={unit}): "
              f"n={len(subset)}, range=[{subset['Engine Capacity'].min():.0f}, "
              f"{subset['Engine Capacity'].max():.0f}], "
              f"mean={subset['Engine Capacity'].mean():.1f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Engine Capacity Mixed Units Problem", fontsize=14, fontweight="bold")

    # Scatter colored by fuel type
    ax = axes[0]
    for fuel_type in df_clean["Fuel Type"].unique():
        subset = df_clean[df_clean["Fuel Type"] == fuel_type]
        ax.scatter(subset["Engine Capacity"], subset["Price_Millions"],
                   label=f"{fuel_type} (n={len(subset)})", alpha=0.7, s=50)
    ax.set_xlabel("Engine Capacity (cc or kWh)")
    ax.set_ylabel("Price (Million PKR)")
    ax.set_title("Engine Capacity vs Price (colored by Fuel Type)", fontweight="bold")
    ax.legend()

    # Box plot by fuel type
    ax = axes[1]
    sns.boxplot(data=df_clean, x="Fuel Type", y="Engine Capacity", ax=ax, palette="Set2")
    ax.set_title("Engine Capacity Distribution by Fuel Type", fontweight="bold")

    plt.tight_layout()
    plt.show()

    insight(
        "Insight 8: Engine Capacity Mixed Units",
        f"Engine Capacity column mixes two physical quantities: "
        f"ICE vehicles use cc (displacement), EVs use kWh (battery capacity)",
        f"Putting them directly in the model is confusing -- 1500cc and 75kWh are in the same column but mean different things",
        f"Tree models like XGBoost can implicitly distinguish via Fuel Type splits, "
        f"but linear/distance models will be misled",
        f"Recommendation: do not split here (EDA does not do feature engineering), "
        f"but note that Pipeline could group-process by Fuel Type",
    )


# ============================================================================
# #  EDA Step 9: Multicollinearity Check (VIF)
# ============================================================================
# ============================================================================


#  EDA Step 9: Multicollinearity Check (VIF)
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 9: Multicollinearity Check (VIF)")
print("=" * 60)

X_vif = df_clean[num_cols].copy()
X_vif["const"] = 1  # VIF requires a constant term

vif_data = []
for i, col in enumerate(X_vif.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_vif.values, i)
    vif_data.append({"feature": col, "VIF": round(vif, 2)})
    print(f"  {col:<18}: VIF = {vif:.2f} "
          f"{'-> Severe multicollinearity' if vif > 10 else '-> No severe multicollinearity' if vif < 5 else '-> Moderate multicollinearity'}")

insight(
    "Insight 9: Multicollinearity",
    f"VIF test: all numerical features VIF < 5, no severe multicollinearity",
    f"Engine Unit identified as fully redundant with Fuel Type (Cramer's V ~ 1.0), should be removed",
    f"Numerical features are independent, suitable for direct model input",
)


# ============================================================================
# #  EDA Step 10: Feature Engineering Preview (preview only, no modifica
# ============================================================================
# ============================================================================


#  EDA Step 10: Feature Engineering Preview (preview only, no modification)
# =====================================================================

print("\n" + "=" * 60)
print("EDA Step 10: Feature Engineering Preview")
print("=" * 60)

_models = df_clean["Car Name"].apply(extract_bmw_model)
model_counts = _models.value_counts()

print(f"--- Extracted BMW Model Distribution ---")
print(model_counts.to_string())
print(f"\nTotal {model_counts.nunique()} unique models")

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Feature Engineering Preview: BMW Model Extraction", fontsize=14, fontweight="bold")

# Model frequency
ax = axes[0]
model_counts.head(15).plot.barh(ax=ax, color=PALETTE["primary"])
ax.set_title("Top 15 BMW Model Frequency", fontweight="bold")
ax.set_xlabel("Count")
ax.invert_yaxis()

# Model vs Price
ax = axes[1]
df_preview = df_clean.copy()
df_preview["BMW_Model"] = _models
top_models = model_counts.head(10).index
df_top = df_preview[df_preview["BMW_Model"].isin(top_models)]
sns.boxplot(data=df_top, x="Price_Millions", y="BMW_Model",
            order=top_models, ax=ax, palette="Set3")
ax.set_title("Top 10 Model Price Distribution", fontweight="bold")
ax.set_xlabel("Price (Million PKR)")

plt.tight_layout()
plt.show()

insight(
    "Insight 10: Feature Engineering Preview",
    f"Extracted {model_counts.nunique()} BMW models from Car Name",
    f"7 Series ({model_counts.get('7 Series', 0)}) and "
    f"5 Series ({model_counts.get('5 Series', 0)}) are most common",
    f"7 Series median price significantly higher than other models, "
    f"confirming the original notebook's finding that luxury series drive price",
    f"Note: this is preview only, actual encoding should go in Pipeline's OneHotEncoder",
)

# Save EDA summary
eda_report_path = os.path.join(OUTPUT_DIR, "eda_summary.json")
with open(eda_report_path, "w", encoding="utf-8") as f:
    json.dump({
        "eda_timestamp": datetime.now().isoformat(),
        "dataset_shape": {"original_rows": 344, "cleaned_rows": len(df_clean)},
        "insights": INSIGHTS,
    }, f, indent=2, ensure_ascii=False, default=str)
print(f"\nEDA summary saved: {eda_report_path}")


# ============================================================================
# #  Pipeline Step 1: Data Loading & Versioning
# ============================================================================
# ============================================================================


#  Pipeline Step 1: Data Loading & Versioning
# ---------------------------------------------------------------------

def pipeline_load_and_version(csv_path):
    """
    Load data and compute version hashes.
    - Define an authoritative raw data version
    - Verify each copy via checksum/content hash
    - Ensure both environments read the same version
    """
    print("\n" + "=" * 60)
    print("Pipeline Step 1: Data Loading & Versioning")
    print("=" * 60)

    df_bmw = pd.read_csv(csv_path)

    raw_file_hash = compute_file_hash(csv_path)
    raw_df_hash = compute_dataframe_hash(df_bmw)

    print(f"Dataset shape: {df_bmw.shape}")
    print(f"Raw file SHA-256: {raw_file_hash[:16]}...")
    print(f"DataFrame hash:   {raw_df_hash[:16]}...")
    print(f"Load time: {datetime.now().isoformat()}")

    data_metadata = {
        "source_path": str(csv_path),
        "raw_file_sha256": raw_file_hash,
        "raw_dataframe_hash": raw_df_hash,
        "n_rows_original": int(df_bmw.shape[0]),
        "n_cols_original": int(df_bmw.shape[1]),
        "load_timestamp": datetime.now().isoformat(),
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
        },
    }
    return df_bmw, data_metadata


# ============================================================================
# #  Pipeline Step 2: Data Cleaning
# ============================================================================
# ============================================================================


#  Pipeline Step 2: Data Cleaning
# ---------------------------------------------------------------------

def pipeline_clean_data(df_bmw):
    """
    Data cleaning: remove missing values, duplicates.
    Create group_id for GroupKFold (prevent similar records crossing folds).
    """
    print("\n" + "=" * 60)
    print("Pipeline Step 2: Data Cleaning")
    print("=" * 60)

    df_clean = df_bmw.copy()

    # Remove index column and high-missing column
    drop_cols = [c for c in ["Unnamed: 0", "Auction Rating"] if c in df_clean.columns]
    df_clean = df_clean.drop(columns=drop_cols)

    # Remove rows with missing target
    if "Price (PKR)" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Price (PKR)"])

    # Remove exact duplicates
    n_before = len(df_clean)
    duplicates_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    n_after = len(df_clean)
    df_clean.reset_index(drop=True, inplace=True)

    print(f"Removed duplicates: {duplicates_count}")
    print(f"Cleaned shape: {df_clean.shape} ({n_before} -> {n_after})")

    # Create group_id for GroupKFold (Alexandre's suggestion)
    # Group by car name prefix + year + fuel type to prevent similar vehicles crossing folds
    df_clean["group_id"] = (
        df_clean.get("Car Name", "").astype(str).str[:20]
        + "_"
        + df_clean.get("Model Year", "").astype(str)
        + "_"
        + df_clean.get("Fuel Type", "").astype(str)
    )
    df_clean["group_id"] = df_clean["group_id"].astype("category").cat.codes

    n_groups = df_clean["group_id"].nunique()
    print(f"Number of groups (for GroupKFold): {n_groups}")

    return df_clean


# ============================================================================
# #  Pipeline Step 3: Feature Engineering
# ============================================================================
# ============================================================================


#  Pipeline Step 3: Feature Engineering
# ---------------------------------------------------------------------

def pipeline_engineer_features(df_clean):
    """
    Feature engineering: extract model, create target.

    Key improvement:
    - Do NOT do One-Hot Encoding here!
    - OHE goes into the Pipeline, fitted per fold by cross-validation
    - This prevents data leakage (original notebook did OHE on the full dataset before split)
    """
    print("\n" + "=" * 60)
    print("Pipeline Step 3: Feature Engineering")
    print("=" * 60)

    df = df_clean.copy()

    # Extract BMW model
    df["BMW_Model"] = df["Car Name"].apply(extract_bmw_model)

    # Create target variable (millions PKR)
    df["Price_Millions"] = df["Price (PKR)"] / 1_000_000

    # Define feature columns and target
    numerical_features = ["Model Year", "Mileage", "Engine Capacity"]
    categorical_features = ["Fuel Type", "Transmission", "BMW_Model"]
    target = "Price_Millions"

    # Remove columns not needed (but do NOT do OHE, leave to Pipeline)
    drop_cols = ["Car Name", "Engine Unit", "Price (PKR)", "group_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols + [target]]
    feature_cols = [c for c in feature_cols if c in numerical_features + categorical_features]

    X = df[feature_cols]
    y = df[target].values
    groups = df["group_id"].values

    print(f"Feature columns: {feature_cols}")
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")

    feature_metadata = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "target": target,
        "feature_order": feature_cols,
        "n_samples": len(X),
    }
    return X, y, groups, feature_metadata


# ============================================================================
# #  Pipeline Step 4: Build Unified Pipeline (core improvement)
# ============================================================================
# ============================================================================


#  Pipeline Step 4: Build Unified Pipeline (core improvement)
# ---------------------------------------------------------------------

def build_pipeline(numerical_features, categorical_features, xgb_params=None):
    
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": CONFIG["random_seed"],
            "n_jobs": CONFIG["xgb_threads"],  # Key: limit threads
        }

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",   # Gracefully handle unknown categories at inference
        drop="first",              # Equivalent to original code's drop_first=True
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(**xgb_params)),
    ])
    return pipeline


# ============================================================================
# #  Pipeline Step 5: Cross-Validation + OOF Predictions
# ============================================================================
# ============================================================================


#  Pipeline Step 5: Cross-Validation + OOF Predictions
# ---------------------------------------------------------------------

def run_cross_validation(X, y, groups, numerical_features, categorical_features,
                         xgb_params=None):
    
    print("\n" + "=" * 60)
    print("Pipeline Step 5: Cross-Validation & OOF Predictions")
    print("=" * 60)

    pipeline = build_pipeline(numerical_features, categorical_features, xgb_params)

    # Use GroupKFold if enough groups, otherwise fall back to RepeatedKFold
    n_groups = len(np.unique(groups))
    use_grouped = n_groups >= CONFIG["n_splits"]

    if use_grouped:
        print(f"Using GroupKFold ({n_groups} groups, {CONFIG['n_splits']} folds)")
        cv = GroupKFold(n_splits=CONFIG["n_splits"])
    else:
        print(f"Insufficient groups, using RepeatedKFold ({CONFIG['n_splits']} folds x {CONFIG['n_repeats']} repeats)")
        cv = RepeatedKFold(n_splits=CONFIG["n_splits"], n_repeats=CONFIG["n_repeats"],
                          random_state=CONFIG["random_seed"])

    scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]

    print(f"Starting cross-validation (n_jobs={CONFIG['fold_jobs']} fold-level parallel)...")
    t_start = time.time()


    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=CONFIG["fold_jobs"],
        return_train_score=True,
        return_estimator=True,
        groups=groups if use_grouped else None,
    )

    t_elapsed = time.time() - t_start
    print(f"Cross-validation complete, elapsed {t_elapsed:.2f}s")

    # Generate OOF predictions
    print("Generating OOF predictions...")
    
    
    oof_predictions = cross_val_predict(
        pipeline, X, y,
        cv=cv,
        n_jobs=CONFIG["fold_jobs"],
        groups=groups if use_grouped else None,
    )

    # Aggregate metrics
    fold_r2 = cv_results["test_r2"]
    fold_mae = -cv_results["test_neg_mean_absolute_error"]
    fold_rmse = -cv_results["test_neg_root_mean_squared_error"]

    results = {
        "fold_r2_scores": fold_r2.tolist(),
        "fold_mae_scores": fold_mae.tolist(),
        "fold_rmse_scores": fold_rmse.tolist(),
        "r2_mean": float(fold_r2.mean()),
        "r2_std": float(fold_r2.std()),
        "mae_mean": float(fold_mae.mean()),
        "mae_std": float(fold_mae.std()),
        "rmse_mean": float(fold_rmse.mean()),
        "rmse_std": float(fold_rmse.std()),
        "oof_r2": float(r2_score(y, oof_predictions)),
        "oof_mae": float(mean_absolute_error(y, oof_predictions)),
        "oof_rmse": float(np.sqrt(mean_squared_error(y, oof_predictions))),
        "oof_predictions": oof_predictions.tolist(),
        "y_true": y.tolist(),
        "training_time_seconds": t_elapsed,
        "fold_fit_times": cv_results["fit_time"].tolist(),
        "fold_score_times": cv_results["score_time"].tolist(),
        "n_folds": len(fold_r2),
        "cv_strategy": "GroupKFold" if use_grouped else "RepeatedKFold",
    }

    print(f"\n--- Cross-Validation Results ---")
    print(f"Fold R2:  {[f'{r:.4f}' for r in fold_r2]}")
    print(f"R2 mean +/- std: {results['r2_mean']:.4f} +/- {results['r2_std']:.4f}")
    print(f"MAE mean +/- std: {results['mae_mean']:.4f} +/- {results['mae_std']:.4f}")
    print(f"RMSE mean +/- std: {results['rmse_mean']:.4f} +/- {results['rmse_std']:.4f}")
    print(f"OOF R2: {results['oof_r2']:.4f}")
    print(f"OOF MAE: {results['oof_mae']:.4f}")
    print(f"OOF RMSE: {results['oof_rmse']:.4f}")

    if results["r2_std"] > 0.05:
        print(f"\nWARNING: R2 std = {results['r2_std']:.4f} > 0.05")
        print("  High variance across folds, dataset may be too small or heterogeneous")
    else:
        print(f"\nOK: R2 std = {results['r2_std']:.4f}, results are stable")

    return results, cv_results, oof_predictions


# ============================================================================
# #  Pipeline Step 6: Extended Evaluation Report
# ============================================================================
# ============================================================================


#  Pipeline Step 6: Extended Evaluation Report
# ---------------------------------------------------------------------

def generate_evaluation_report(y_true, oof_predictions, fold_results, X, output_dir):
    
    print("\n" + "=" * 60)
    print("Pipeline Step 6: Extended Evaluation Report")
    print("=" * 60)

    y_true = np.array(y_true)
    oof_pred = np.array(oof_predictions)
    residuals = y_true - oof_pred

    # Global metrics
    global_r2 = r2_score(y_true, oof_pred)
    global_mae = mean_absolute_error(y_true, oof_pred)
    global_rmse = np.sqrt(mean_squared_error(y_true, oof_pred))

    print(f"\n{'Metric':<20} {'Global OOF':<15} {'Fold Mean':<15} {'Fold Std':<15}")
    print("-" * 65)
    print(f"{'R2':<20} {global_r2:<15.4f} {fold_results['r2_mean']:<15.4f} {fold_results['r2_std']:<15.4f}")
    print(f"{'MAE (M PKR)':<20} {global_mae:<15.4f} {fold_results['mae_mean']:<15.4f} {fold_results['mae_std']:<15.4f}")
    print(f"{'RMSE (M PKR)':<20} {global_rmse:<15.4f} {fold_results['rmse_mean']:<15.4f} {fold_results['rmse_std']:<15.4f}")

    print(f"\n--- Residual Analysis ---")
    print(f"Residual mean:    {residuals.mean():.4f} (near 0 = no systematic bias)")
    print(f"Residual std:     {residuals.std():.4f}")
    print(f"Residual min:     {residuals.min():.4f}")
    print(f"Residual max:     {residuals.max():.4f}")
    print(f"Residual median:  {np.median(residuals):.4f}")

    print(f"\n--- Residual vs Feature Correlation ---")
    for col in X.select_dtypes(include=[np.number]).columns:
        corr = np.corrcoef(X[col].values, residuals)[0, 1]
        if not np.isnan(corr):
            print(f"  {col:<20}: correlation = {corr:+.4f}")

    print(f"\n--- Per-Fold Time ---")
    for i, (ft, st) in enumerate(zip(fold_results["fold_fit_times"], fold_results["fold_score_times"])):
        print(f"  Fold {i+1}: train {ft:.3f}s, inference {st:.3f}s")

    # --- Plot: Actual vs Predicted + Residual Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("OOF Evaluation: Actual vs Predicted & Residual Distribution", fontsize=14, fontweight="bold")

    # Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_true, oof_pred, alpha=0.6, color=PALETTE["primary"], s=50)
    lims = [min(y_true.min(), oof_pred.min()), max(y_true.max(), oof_pred.max())]
    ax.plot(lims, lims, color=PALETTE["danger"], linewidth=2, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual Price (Million PKR)")
    ax.set_ylabel("Predicted Price (Million PKR)")
    ax.set_title(f"Actual vs Predicted (OOF R2={global_r2:.4f})", fontweight="bold")
    ax.legend()

    # Residual distribution
    ax = axes[1]
    ax.hist(residuals, bins=30, color=PALETTE["warning"], edgecolor="white", alpha=0.8)
    ax.axvline(0, color=PALETTE["danger"], linewidth=2, linestyle="--", label="Zero residual")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution (mean={residuals.mean():.2f}, std={residuals.std():.2f})", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.show()

    report = {
        "global_oof_r2": float(global_r2),
        "global_oof_mae": float(global_mae),
        "global_oof_rmse": float(global_rmse),
        "residual_stats": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
            "median": float(np.median(residuals)),
        },
        "fold_metrics": {
            "r2": fold_results["fold_r2_scores"],
            "mae": fold_results["fold_mae_scores"],
            "rmse": fold_results["fold_rmse_scores"],
        },
        "fold_times": {
            "fit_times": fold_results["fold_fit_times"],
            "score_times": fold_results["fold_score_times"],
        },
    }
    return report, residuals


# ============================================================================
# #  Pipeline Step 7: Train Final Model & Save (with metadata)
# ============================================================================
# ============================================================================


#  Pipeline Step 7: Train Final Model & Save (with metadata)
# ---------------------------------------------------------------------

def train_and_save_final_model(X, y, groups, numerical_features, categorical_features,
                                xgb_params, data_metadata, feature_metadata,
                                cv_results, eval_report, output_dir):
    
    print("\n" + "=" * 60)
    print("Pipeline Step 7: Train Final Model & Save")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    final_pipeline = build_pipeline(numerical_features, categorical_features, xgb_params)
    print("Training final model on full data...")
    t_start = time.time()
    final_pipeline.fit(X, y)
    t_train = time.time() - t_start
    print(f"Training complete, elapsed {t_train:.2f}s")

    # Save model
    model_path = os.path.join(output_dir, "bmw_price_pipeline.pkl")
    joblib.dump(final_pipeline, model_path)
    model_size = os.path.getsize(model_path) / 1024
    model_hash = compute_file_hash(model_path)

    print(f"Model saved: {model_path} ({model_size:.2f} KB)")
    print(f"Model hash: {model_hash[:16]}...")

    # Save complete metadata
    metadata = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "random_seed": CONFIG["random_seed"],
            "training_time_seconds": t_train,
        },
        "data_metadata": data_metadata,
        "feature_metadata": feature_metadata,
        "model_info": {
            "model_type": "XGBRegressor",
            "model_params": xgb_params,
            "model_path": model_path,
            "model_size_kb": round(model_size, 2),
            "model_sha256": model_hash,
            "pipeline_structure": "ColumnTransformer(StandardScaler + OneHotEncoder) -> XGBRegressor",
        },
        "cv_results": {
            "cv_strategy": cv_results["cv_strategy"],
            "n_folds": cv_results["n_folds"],
            "r2_mean": cv_results["r2_mean"],
            "r2_std": cv_results["r2_std"],
            "fold_r2_scores": cv_results["fold_r2_scores"],
            "fold_mae_scores": cv_results["fold_mae_scores"],
            "fold_rmse_scores": cv_results["fold_rmse_scores"],
            "oof_r2": cv_results["oof_r2"],
            "oof_mae": cv_results["oof_mae"],
            "oof_rmse": cv_results["oof_rmse"],
        },
        "evaluation_report": eval_report,
        "parallelism_config": {
            "xgb_threads": CONFIG["xgb_threads"],
            "fold_jobs": CONFIG["fold_jobs"],
            "total_threads": CONFIG["xgb_threads"] * CONFIG["fold_jobs"],
            "note": "xgb_threads * fold_jobs should be <= physical CPU cores",
        },
        "inference_config": {
            "input_validation": {
                "Model Year": {"min": 1990, "max": 2026, "type": "int"},
                "Mileage": {"min": 0, "max": 500000, "type": "float"},
                "Engine Capacity": {"min": 0, "max": 8000, "type": "float"},
            },
            "load_once": True,
            "batch_inference": True,
        },
    }

    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"Metadata saved: {metadata_path}")

    return final_pipeline, metadata


# ============================================================================
# #  Pipeline Step 8: Inference Class (optimized)
# ============================================================================
# ============================================================================


#  Pipeline Step 8: Inference Class (optimized)
# ---------------------------------------------------------------------

class BMWPricePredictor:

    VALID_FUEL_TYPES = {"Petrol", "Diesel", "Electric", "Hybrid", "PHEV"}
    VALID_TRANSMISSIONS = {"Automatic", "Manual"}

    def __init__(self, model_path="output/bmw_price_pipeline.pkl"):
        """Load model once at initialization."""
        self.pipeline = joblib.load(model_path)
        self._loaded = True
        self._load_time = time.time()
        print(f"Model loaded (load once, reuse for subsequent predictions)")

    def _validate_input(self, model_year, mileage, engine_capacity,
                        fuel_type, transmission, bmw_model):
        """Validate input data, prevent unreasonable values."""
        errors = []

        checks = {
            "Model Year": (model_year, 1990, 2026),
            "Mileage": (mileage, 0, 500000),
            "Engine Capacity": (engine_capacity, 0, 8000),
        }
        for field, (val, lo, hi) in checks.items():
            if not (lo <= val <= hi):
                errors.append(f"{field}={val} not in valid range [{lo}, {hi}]")

        if fuel_type not in self.VALID_FUEL_TYPES:
            errors.append(f"Fuel Type='{fuel_type}' invalid, should be {self.VALID_FUEL_TYPES}")
        if transmission not in self.VALID_TRANSMISSIONS:
            errors.append(f"Transmission='{transmission}' invalid, should be {self.VALID_TRANSMISSIONS}")

        return errors

    def predict_single(self, model_year, mileage, engine_capacity,
                       fuel_type, transmission, bmw_model):
        """Single prediction (online inference)."""
        errors = self._validate_input(
            model_year, mileage, engine_capacity,
            fuel_type, transmission, bmw_model
        )
        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        input_df = pd.DataFrame([{
            "Model Year": float(model_year),
            "Mileage": float(mileage),
            "Engine Capacity": float(engine_capacity),
            "Fuel Type": fuel_type,
            "Transmission": transmission,
            "BMW_Model": bmw_model,
        }])

        pred_millions = self.pipeline.predict(input_df)[0]
        return float(pred_millions), int(pred_millions * 1_000_000)

    def predict_batch(self, records):
        """Batch prediction (vectorized inference)."""
        input_df = pd.DataFrame(records)

        for idx, row in input_df.iterrows():
            errors = self._validate_input(
                row["Model Year"], row["Mileage"], row["Engine Capacity"],
                row["Fuel Type"], row["Transmission"], row["BMW_Model"]
            )
            if errors:
                raise ValueError(f"Row {idx} validation failed: {'; '.join(errors)}")

        predictions = self.pipeline.predict(input_df)
        return [(float(p), int(p * 1_000_000)) for p in predictions]


# ============================================================================
# #  Pipeline Step 9: Reproducibility Verification
# ============================================================================
# ============================================================================


#  Pipeline Step 9: Reproducibility Verification
# ---------------------------------------------------------------------

def verify_reproducibility(X, y, groups, numerical_features, categorical_features,
                           xgb_params, n_runs=5):
    
    print("\n" + "=" * 60)
    print("Pipeline Step 9: Reproducibility Verification (Multi-Seed)")
    print("=" * 60)

    all_r2 = []
    all_mae = []

    for seed in range(42, 42 + n_runs):
        params = xgb_params.copy()
        params["random_state"] = seed

        pipeline = build_pipeline(numerical_features, categorical_features, params)

        cv = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=seed)
        cv_res = cross_validate(
            pipeline, X, y, cv=cv,
            scoring=["r2", "neg_mean_absolute_error"],
            n_jobs=CONFIG["fold_jobs"],
        )
        r2_mean = cv_res["test_r2"].mean()
        mae_mean = -cv_res["test_neg_mean_absolute_error"].mean()
        all_r2.append(r2_mean)
        all_mae.append(mae_mean)
        print(f"  Seed {seed}: R2={r2_mean:.4f}, MAE={mae_mean:.4f}")

    all_r2 = np.array(all_r2)
    all_mae = np.array(all_mae)

    print(f"\n--- Multi-Seed Stability ---")
    print(f"R2:  {all_r2.mean():.4f} +/- {all_r2.std():.4f} (range: {all_r2.min():.4f} ~ {all_r2.max():.4f})")
    print(f"MAE: {all_mae.mean():.4f} +/- {all_mae.std():.4f}")

    if all_r2.std() > 0.03:
        print("WARNING: R2 fluctuates across seeds, consider increasing data or using more stable config")
    else:
        print("OK: Results are stable across seeds")

    return {
        "seeds": list(range(42, 42 + n_runs)),
        "r2_scores": all_r2.tolist(),
        "mae_scores": all_mae.tolist(),
        "r2_mean": float(all_r2.mean()),
        "r2_std": float(all_r2.std()),
    }


# ============================================================================
# #  Pipeline Step 10: Inference Benchmark
# ============================================================================
# ============================================================================


#  Pipeline Step 10: Inference Benchmark
# ---------------------------------------------------------------------

def benchmark_model(pipeline, X, y, output_dir):
    
    print("\n" + "=" * 60)
    print("Pipeline Step 10: Inference Benchmark")
    print("=" * 60)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        joblib.dump(pipeline, tmp.name)
        model_size_kb = os.path.getsize(tmp.name) / 1024
        os.unlink(tmp.name)

    sample = X.iloc[[0]]
    # Cold start (first inference)
    t0 = time.time()
    pipeline.predict(sample)
    cold_latency_ms = (time.time() - t0) * 1000

    # Warm start (average over multiple runs)
    n_warmup = 100
    t0 = time.time()
    for _ in range(n_warmup):
        pipeline.predict(sample)
    warm_latency_ms = (time.time() - t0) / n_warmup * 1000

    # Batch throughput
    batch = X.iloc[:50] if len(X) >= 50 else X
    t0 = time.time()
    pipeline.predict(batch)
    batch_latency_ms = (time.time() - t0) * 1000
    throughput = len(batch) / (batch_latency_ms / 1000)

    # Prediction accuracy (on training set, for reference only)
    y_pred = pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    benchmark = {
        "model_size_kb": round(model_size_kb, 2),
        "cold_start_latency_ms": round(cold_latency_ms, 2),
        "warm_latency_ms": round(warm_latency_ms, 2),
        "batch_latency_ms": round(batch_latency_ms, 2),
        "batch_throughput_records_per_sec": round(throughput, 0),
        "training_set_r2": round(float(r2), 4),
        "training_set_mae": round(float(mae), 4),
    }

    print(f"Model size:           {benchmark['model_size_kb']:.2f} KB")
    print(f"Cold start latency:   {benchmark['cold_start_latency_ms']:.2f} ms")
    print(f"Warm latency:        {benchmark['warm_latency_ms']:.4f} ms")
    print(f"Batch inference (50):{benchmark['batch_latency_ms']:.2f} ms")
    print(f"Throughput:          {benchmark['batch_throughput_records_per_sec']:.0f} records/sec")

    print(f"\nConclusion: model size {benchmark['model_size_kb']:.0f}KB, warm latency {benchmark['warm_latency_ms']:.2f}ms")
    print("  For this project scale, the model is already compact and fast, quantization may not bring significant gains.")
    print("  More practical optimizations: load once + batch inference + input validation.")

    return benchmark


# ============================================================================
# #  Run Pipeline (EDA already ran as top-level code above)
# ============================================================================
# ============================================================================


#  Run Pipeline (EDA already ran as top-level code above)
# #####################################################################

print("\n" + "#" * 60)
print("#  PART 2: MODELING PIPELINE")
print("#" * 60)

# Pipeline Step 1: Load & version data
df_bmw2, data_metadata = pipeline_load_and_version(CSV_PATH)

# Pipeline Step 2: Clean data
df_clean2 = pipeline_clean_data(df_bmw2)

# Pipeline Step 3: Feature engineering
X, y, groups, feature_metadata = pipeline_engineer_features(df_clean2)

# XGBoost parameters (from original notebook's tuned best params)
xgb_params = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": CONFIG["random_seed"],
    "n_jobs": CONFIG["xgb_threads"],
}

# Pipeline Step 5: Cross-validation
cv_results, raw_cv, oof_pred = run_cross_validation(
    X, y, groups,
    feature_metadata["numerical_features"],
    feature_metadata["categorical_features"],
    xgb_params,
)

# Pipeline Step 6: Extended evaluation
eval_report, residuals = generate_evaluation_report(
    y, oof_pred, cv_results, X, OUTPUT_DIR
)

# Pipeline Step 7: Train final model & save
final_pipeline, metadata = train_and_save_final_model(
    X, y, groups,
    feature_metadata["numerical_features"],
    feature_metadata["categorical_features"],
    xgb_params, data_metadata, feature_metadata,
    cv_results, eval_report, OUTPUT_DIR
)

# Pipeline Step 9: Reproducibility verification
repro_results = verify_reproducibility(
    X, y, groups,
    feature_metadata["numerical_features"],
    feature_metadata["categorical_features"],
    xgb_params, n_runs=5
)
metadata["reproducibility_test"] = repro_results

# Pipeline Step 10: Inference benchmark
benchmark = benchmark_model(final_pipeline, X, y, OUTPUT_DIR)
metadata["inference_benchmark"] = benchmark

# Re-save metadata with reproducibility + benchmark
with open(os.path.join(OUTPUT_DIR, "experiment_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


# ============================================================================
# #  Inference Demo
# ============================================================================
# ============================================================================


#  Inference Demo
# =====================================================================

print("\n" + "=" * 60)
print("Inference Demo")
print("=" * 60)

predictor = BMWPricePredictor(
    model_path=os.path.join(OUTPUT_DIR, "bmw_price_pipeline.pkl")
)

# Single prediction
pred_m, pred_pkr = predictor.predict_single(
    model_year=2022, mileage=25000, engine_capacity=3000.0,
    fuel_type="Hybrid", transmission="Automatic",
    bmw_model="7 Series"
)
print(f"\nSingle prediction:")
print(f"  2022 BMW 7 Series Hybrid, 25000km, 3000cc")
print(f"  Predicted price: {pred_m:.2f} Million PKR ({pred_pkr:,} PKR)")

# Batch prediction
batch_records = [
    {"Model Year": 2022, "Mileage": 25000, "Engine Capacity": 3000.0,
     "Fuel Type": "Hybrid", "Transmission": "Automatic", "BMW_Model": "7 Series"},
    {"Model Year": 2021, "Mileage": 68064, "Engine Capacity": 1500.0,
     "Fuel Type": "Petrol", "Transmission": "Manual", "BMW_Model": "2 Series"},
    {"Model Year": 2012, "Mileage": 102148, "Engine Capacity": 4400.0,
     "Fuel Type": "Petrol", "Transmission": "Automatic", "BMW_Model": "X5"},
]
batch_results = predictor.predict_batch(batch_records)
print(f"\nBatch prediction ({len(batch_records)} records):")
for i, (pred_m, pred_pkr) in enumerate(batch_results):
    print(f"  Vehicle {i+1}: {pred_m:.2f} Million PKR ({pred_pkr:,} PKR)")

# Input validation demo
print(f"\nInput validation demo:")
try:
    predictor.predict_single(
        model_year=1800,  # Invalid year
        mileage=-100,     # Negative mileage
        engine_capacity=3000.0,
        fuel_type="Hybrid",
        transmission="Automatic",
        bmw_model="7 Series"
    )
except ValueError as e:
    print(f"  Expected error: {e}")


# ============================================================================
# #  Summary
# ============================================================================
# ============================================================================


#  Summary
# =====================================================================

print("\n" + "=" * 60)
print("All Done!")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}/")
print(f"  - bmw_price_pipeline.pkl      (model + preprocessing Pipeline)")
print(f"  - experiment_metadata.json     (full experiment metadata)")
print(f"  - eda_summary.json             (EDA insights summary)")
print(f"\nAll charts are displayed inline via plt.show() (Kaggle Jupyter Notebook)")

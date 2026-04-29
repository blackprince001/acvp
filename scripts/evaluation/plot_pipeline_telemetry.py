#!/usr/bin/env python3
"""Visualize predict-mode telemetry from the live pipeline.

Reads one or more telemetry CSVs produced by ``run_pipeline_live.sh``
(65-column predict-mode format) and generates:

  - prediction_error_<feature>.png    : rolling RMSE of T+1 prediction
  - horizon_fan_<feature>.png         : actual + T+1..T+5 fan band

Usage:
  uv run python scripts/evaluation/plot_pipeline_telemetry.py \\
    --csv experiments/inference/pipeline_live/*/telemetry.csv \\
    --output-dir experiments/inference/pipeline_live/figures

  # Compare multiple estimators side-by-side (labels inferred from path):
  uv run python scripts/evaluation/plot_pipeline_telemetry.py \\
    --csv experiments/inference/pipeline_live/20260429_032049_yolov8n_tcn/telemetry.csv \\
         experiments/inference/pipeline_live/20260429_030928_yolov8n_lstm/telemetry.csv \\
         experiments/inference/pipeline_live/20260429_031624_yolov8n_gru/telemetry.csv \\
    --output-dir experiments/inference/pipeline_live/figures
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import FEATURE_NAMES  # noqa: E402

HEADLINE_FEATURES = ["vehicle_count", "occupancy_ratio", "congestion_index"]
sns.set_theme(style="whitegrid", context="paper")
PALETTE = {"lstm": "#1f77b4", "gru": "#2ca02c", "tcn": "#d62728"}
PALETTE_LIST = list(PALETTE.values()) + ["#9467bd", "#ff7f0e", "#8c564b"]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Plot predict-mode pipeline telemetry.")
  p.add_argument("--csv", required=True, nargs="+", help="Telemetry CSV(s) from run_pipeline_live.sh.")
  p.add_argument("--output-dir", required=True)
  p.add_argument("--features", nargs="*", default=None, help="Features to plot (default: headline set).")
  p.add_argument("--window", type=int, default=600, help="Frames to show in time-series plots (0 = all).")
  p.add_argument("--rolling-window", type=int, default=50, help="Rolling window size for error plots.")
  return p.parse_args()


def infer_label(csv_path: str) -> str:
  """Extract a short label from the telemetry path, e.g. 'yolov8n_tcn'."""
  parent = Path(csv_path).parent.name
  m = re.search(r"(yolo\S+|rtdetr\S+)_(lstm|gru|tcn)", parent, re.IGNORECASE)
  if m:
    return m.group(0)
  return parent


def load_telemetry(csv_path: str) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  # The CSV has duplicate column names (vehicle_count appears at col 3 and 5).
  # Pandas appends .1 to duplicates. The feature-vector columns (5-14) are the
  # ones we want as "actual"; the first two (3-4) are raw filtered duplicates.
  return df


def get_actual(df: pd.DataFrame, feature: str) -> pd.Series:
  """Get the actual feature column from the feature-vector block."""
  # Feature vector starts after frame_idx, timestamp, vehicle_count, occupancy_ratio
  # i.e. columns 4..13 (0-indexed) map to FEATURE_NAMES[0..9].
  # With duplicate names, pandas renames the second vehicle_count → vehicle_count.1, etc.
  # Try the .1 suffix first (the feature-vector copy), fall back to original.
  for suffix in [".1", ""]:
    col = feature + suffix
    if col in df.columns:
      return pd.to_numeric(df[col], errors="coerce")
  raise KeyError(f"Feature {feature} not found in CSV columns: {list(df.columns)}")


def get_pred(df: pd.DataFrame, feature: str, horizon: int = 1) -> pd.Series:
  col = f"pred_h{horizon}_{feature}"
  if col not in df.columns:
    raise KeyError(f"Prediction column {col} not found")
  return pd.to_numeric(df[col], errors="coerce")


def plot_prediction_error(runs: dict[str, pd.DataFrame], feature: str, out_dir: Path, rolling: int):
  fig, ax = plt.subplots(figsize=(10, 3.0))

  for i, (label, df) in enumerate(runs.items()):
    actual = get_actual(df, feature)
    pred = get_pred(df, feature, horizon=1)
    err = (actual - pred) ** 2
    rolling_rmse = err.rolling(rolling, min_periods=1).mean().apply(np.sqrt)
    color = PALETTE.get(label.split("_")[-1], PALETTE_LIST[i % len(PALETTE_LIST)])
    ax.plot(rolling_rmse, color=color, lw=1.0, alpha=0.85, label=label)

  ax.set_xlabel("frame")
  ax.set_ylabel(f"rolling RMSE (w={rolling})")
  ax.set_title(f"Prediction error over time — {feature}")
  ax.legend(loc="upper right", frameon=False, fontsize=7)
  fig.tight_layout()
  fig.savefig(out_dir / f"prediction_error_{feature}.png", dpi=200)
  plt.close(fig)


def plot_horizon_fan(runs: dict[str, pd.DataFrame], feature: str, out_dir: Path, window: int):
  """Plot actual line with T+1..T+5 predictions as a widening band for each run."""
  n_runs = len(runs)
  fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 3.2), squeeze=False, sharey=True)

  for idx, (label, df) in enumerate(runs.items()):
    ax = axes[0, idx]
    actual = get_actual(df, feature)
    n = len(actual) if window == 0 else min(window, len(actual))
    x = np.arange(n)
    ax.plot(x, actual.iloc[:n], color="black", lw=1.2, label="actual", zorder=10)

    # Collect horizons
    horizons = []
    for h in range(1, 6):
      try:
        horizons.append(get_pred(df, feature, horizon=h).iloc[:n].to_numpy())
      except KeyError:
        break

    if horizons:
      base_color = PALETTE.get(label.split("_")[-1], PALETTE_LIST[idx % len(PALETTE_LIST)])
      # Fan: shade between min/max across horizons, darkest at T+1
      h_stack = np.column_stack(horizons)  # (n, H)
      for h in range(len(horizons)):
        alpha = 0.35 - h * 0.05
        ax.fill_between(x, h_stack[:, h], actual.iloc[:n], alpha=max(alpha, 0.08), color=base_color)
      ax.plot(x, horizons[0], color=base_color, lw=0.8, ls="--", label="T+1", alpha=0.9)
      if len(horizons) >= 5:
        ax.plot(x, horizons[4], color=base_color, lw=0.6, ls=":", label="T+5", alpha=0.7)

    ax.set_xlabel("frame")
    ax.set_title(label, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=7)

  axes[0, 0].set_ylabel(feature)
  fig.suptitle(f"Horizon fan — {feature}", fontsize=11)
  fig.tight_layout()
  fig.savefig(out_dir / f"horizon_fan_{feature}.png", dpi=200)
  plt.close(fig)


def main() -> int:
  args = parse_args()
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  features = args.features or HEADLINE_FEATURES

  runs: dict[str, pd.DataFrame] = {}
  for csv_path in args.csv:
    label = infer_label(csv_path)
    runs[label] = load_telemetry(csv_path)
    print(f"Loaded {label}: {len(runs[label])} frames from {csv_path}")

  for feat in features:
    print(f"Plotting {feat}...")
    plot_prediction_error(runs, feat, out_dir, args.rolling_window)
    plot_horizon_fan(runs, feat, out_dir, args.window)

  print(f"\nDone. Figures in {out_dir}/")
  return 0


if __name__ == "__main__":
  sys.exit(main())

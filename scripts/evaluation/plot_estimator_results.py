#!/usr/bin/env python3
"""Generate paper figures from evaluated estimators.

Reads ``predictions.npz`` and ``metrics.json`` from each of
``<root>/{lstm,gru,tcn}/`` and writes plots + a summary table to
``<output_dir>``.

Figures produced:
  - predicted_vs_actual_<feature>.png  : time-series, three models overlaid
  - rmse_per_horizon.png               : per-step RMSE across the horizon
  - rmse_per_feature.png               : per-feature RMSE bar chart
  - scatter_<feature>.png              : prediction scatter (3 panels)
  - metrics_summary.csv                : flat table for the paper
"""

from __future__ import annotations

import argparse
import json
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

MODELS = ["lstm", "gru", "tcn"]
HEADLINE_FEATURES = ["vehicle_count", "occupancy_ratio", "congestion_index"]
sns.set_theme(style="whitegrid", context="paper")
PALETTE = {"lstm": "#1f77b4", "gru": "#2ca02c", "tcn": "#d62728"}


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--root", required=True, help="Root with {lstm,gru,tcn}/ subdirs.")
  p.add_argument("--output-dir", required=True)
  p.add_argument(
    "--time-window",
    type=int,
    default=600,
    help="Number of frames to plot in the time-series view (head of eval).",
  )
  p.add_argument(
    "--horizon-step",
    type=int,
    default=0,
    help="Which horizon step to plot in time-series and scatter (0 = T+1).",
  )
  return p.parse_args()


def load_runs(root: Path) -> dict[str, dict]:
  runs = {}
  for m in MODELS:
    npz = root / m / "predictions.npz"
    js = root / m / "metrics.json"
    if not npz.exists() or not js.exists():
      print(f"[skip] {m}: missing predictions.npz or metrics.json", file=sys.stderr)
      continue
    data = np.load(npz)
    with open(js) as fh:
      metrics = json.load(fh)
    runs[m] = {
      "y_true": data["y_true"],   # (N, H, F)
      "y_pred": data["y_pred"],   # (N, H, F)
      "metrics": metrics,
    }
  if not runs:
    raise SystemExit("No usable runs found.")
  return runs


def plot_time_series(runs, out_dir, feature, h_step, n_points):
  fi = FEATURE_NAMES.index(feature)
  fig, ax = plt.subplots(figsize=(9, 3.2))
  any_run = next(iter(runs.values()))
  truth = any_run["y_true"][:n_points, h_step, fi]
  ax.plot(truth, color="black", lw=1.4, label="actual", alpha=0.85)
  for m, run in runs.items():
    pred = run["y_pred"][:n_points, h_step, fi]
    ax.plot(pred, color=PALETTE[m], lw=1.0, alpha=0.85, label=m.upper())
  ax.set_xlabel("frame (eval set, T+%d)" % (h_step + 1))
  ax.set_ylabel(feature)
  ax.legend(loc="upper right", ncol=4, frameon=False)
  ax.set_title(f"Predicted vs. actual — {feature}")
  fig.tight_layout()
  out = out_dir / f"predicted_vs_actual_{feature}.png"
  fig.savefig(out, dpi=200)
  plt.close(fig)
  print(f"  wrote {out}")


def plot_rmse_per_horizon(runs, out_dir):
  rows = []
  for m, run in runs.items():
    for h, step in enumerate(run["metrics"]["per_horizon_step_original"]):
      rows.append({"model": m.upper(), "horizon_step": h + 1, "rmse": step["rmse"]})
  df = pd.DataFrame(rows)
  fig, ax = plt.subplots(figsize=(6, 3.4))
  for m in MODELS:
    if m.upper() not in df["model"].unique():
      continue
    sub = df[df.model == m.upper()]
    ax.plot(sub.horizon_step, sub.rmse, marker="o", color=PALETTE[m], label=m.upper())
  ax.set_xlabel("horizon step (frames ahead)")
  ax.set_ylabel("RMSE (original units)")
  ax.set_title("Forecast error vs. horizon")
  ax.legend(frameon=False)
  fig.tight_layout()
  out = out_dir / "rmse_per_horizon.png"
  fig.savefig(out, dpi=200)
  plt.close(fig)
  print(f"  wrote {out}")


def plot_rmse_per_feature(runs, out_dir):
  rows = []
  for m, run in runs.items():
    pf = run["metrics"]["per_feature_original"]
    for name, vals in pf.items():
      rows.append({"model": m.upper(), "feature": name, "rmse": vals["rmse"]})
  df = pd.DataFrame(rows)
  fig, ax = plt.subplots(figsize=(9, 3.6))
  sns.barplot(
    data=df,
    x="feature",
    y="rmse",
    hue="model",
    palette={m.upper(): c for m, c in PALETTE.items()},
    ax=ax,
  )
  ax.set_yscale("log")
  ax.set_ylabel("RMSE (log scale)")
  ax.set_title("Per-feature forecast RMSE")
  for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha("right")
  ax.legend(frameon=False, title="")
  fig.tight_layout()
  out = out_dir / "rmse_per_feature.png"
  fig.savefig(out, dpi=200)
  plt.close(fig)
  print(f"  wrote {out}")


def plot_scatter(runs, out_dir, feature, h_step):
  fi = FEATURE_NAMES.index(feature)
  fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.2), sharex=True, sharey=True)
  if len(runs) == 1:
    axes = [axes]
  for ax, (m, run) in zip(axes, runs.items()):
    yt = run["y_true"][:, h_step, fi]
    yp = run["y_pred"][:, h_step, fi]
    ax.scatter(yt, yp, s=4, alpha=0.15, color=PALETTE[m])
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
    r2 = run["metrics"]["per_horizon_step_original"][h_step]["r2"]
    ax.set_title(f"{m.upper()} — R²={r2:.3f}")
    ax.set_xlabel("actual")
  axes[0].set_ylabel("predicted")
  fig.suptitle(f"Prediction scatter — {feature} (T+{h_step + 1})")
  fig.tight_layout()
  out = out_dir / f"scatter_{feature}.png"
  fig.savefig(out, dpi=200)
  plt.close(fig)
  print(f"  wrote {out}")


def write_summary_csv(runs, out_dir):
  rows = []
  for m, run in runs.items():
    o = run["metrics"]["overall_original"]
    row = {
      "model": m.upper(),
      "n_windows": run["metrics"]["n_windows"],
      "input_len": run["metrics"]["input_len"],
      "horizon": run["metrics"]["horizon"],
      "MSE": round(o["mse"], 4),
      "MAE": round(o["mae"], 4),
      "RMSE": round(o["rmse"], 4),
      "R2": round(o["r2"], 4),
    }
    for h, step in enumerate(run["metrics"]["per_horizon_step_original"]):
      row[f"RMSE_T+{h + 1}"] = round(step["rmse"], 4)
    rows.append(row)
  df = pd.DataFrame(rows)
  out = out_dir / "metrics_summary.csv"
  df.to_csv(out, index=False)
  print(f"  wrote {out}")
  return df


def main() -> int:
  args = parse_args()
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  runs = load_runs(Path(args.root))

  print(f"Plotting {len(runs)} estimator(s) → {out_dir}")
  for feat in HEADLINE_FEATURES:
    plot_time_series(runs, out_dir, feat, args.horizon_step, args.time_window)
    plot_scatter(runs, out_dir, feat, args.horizon_step)
  plot_rmse_per_horizon(runs, out_dir)
  plot_rmse_per_feature(runs, out_dir)
  df = write_summary_csv(runs, out_dir)

  print("\nSummary:")
  print(df.to_string(index=False))
  return 0


if __name__ == "__main__":
  sys.exit(main())

"""
Paper-quality plots of detection training runs, grouped across models.

Reads per-epoch ``results.csv`` files written by Ultralytics (YOLO / RT-DETR)
under ``experiments/detection/<model>/results.csv`` and produces overlay plots
where each line is a model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.evaluation.visualization import (  # noqa: E402
  _color_for_model,
  _family_of,
  _save,
  _set_paper_style,
)

# Map heterogeneous loss column names to a generic role.
# YOLO uses box/dfl, RT-DETR uses giou/l1. cls is shared.
_LOSS_ROLE_COLUMNS: dict[str, tuple[str, ...]] = {
  "regression": ("box_loss", "giou_loss"),
  "classification": ("cls_loss",),
  "auxiliary": ("dfl_loss", "l1_loss"),
}

_METRIC_COLS = {
  "map50": "metrics/mAP50(B)",
  "map50_95": "metrics/mAP50-95(B)",
  "precision": "metrics/precision(B)",
  "recall": "metrics/recall(B)",
}


def _ema(values: np.ndarray, alpha: float = 0.6) -> np.ndarray:
  if len(values) == 0:
    return values
  out = np.empty_like(values, dtype=float)
  out[0] = values[0]
  for i in range(1, len(values)):
    out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
  return out


def _resolve_loss_column(df: pd.DataFrame, split: str, role: str) -> str | None:
  for suffix in _LOSS_ROLE_COLUMNS[role]:
    col = f"{split}/{suffix}"
    if col in df.columns:
      return col
  return None


def load_runs(experiments_dir: str | Path) -> dict[str, pd.DataFrame]:
  """Discover ``<model>/results.csv`` directories and return name → DataFrame."""
  base = Path(experiments_dir)
  runs: dict[str, pd.DataFrame] = {}
  for csv in sorted(base.glob("*/results.csv")):
    model = csv.parent.name
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    runs[model] = df
  if not runs:
    raise RuntimeError(f"No results.csv files found under {base}")
  return runs


def _ordered_models(runs: dict[str, pd.DataFrame]) -> list[str]:
  return sorted(runs.keys(), key=lambda m: (_family_of(m), m))


def _plot_overlay(
  ax,
  runs: dict[str, pd.DataFrame],
  column: str,
  *,
  smooth: bool = True,
) -> None:
  for i, model in enumerate(_ordered_models(runs)):
    df = runs[model]
    if column not in df.columns:
      continue
    color = _color_for_model(model, fallback_index=i)
    x = df["epoch"].to_numpy()
    y = df[column].to_numpy(dtype=float)
    if smooth and len(y) > 2:
      ax.plot(x, y, color=color, alpha=0.18, linewidth=0.9, zorder=2)
      ax.plot(x, _ema(y), color=color, alpha=0.95, linewidth=1.4, zorder=3, label=model)
    else:
      ax.plot(x, y, color=color, alpha=0.95, linewidth=1.4, zorder=3, label=model)


def _legend_handles(runs: dict[str, pd.DataFrame]) -> list[Line2D]:
  handles = []
  for i, model in enumerate(_ordered_models(runs)):
    handles.append(
      Line2D(
        [0],
        [0],
        color=_color_for_model(model, fallback_index=i),
        linewidth=1.6,
        label=model,
      )
    )
  return handles


def plot_metric_curves(
  runs: dict[str, pd.DataFrame],
  output: str | Path,
) -> Path:
  """2×2 small multiples: mAP50, mAP50-95, precision, recall over epochs."""
  _set_paper_style()
  fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=False)
  panels = [
    ("map50", "mAP@0.5"),
    ("map50_95", "mAP@0.5:0.95"),
    ("precision", "Precision"),
    ("recall", "Recall"),
  ]
  for ax, (key, label) in zip(axes.flat, panels, strict=True):
    _plot_overlay(ax, runs, _METRIC_COLS[key])
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25, linewidth=0.5)

  fig.suptitle("Validation metrics across training", fontsize=11, y=1.0)
  fig.legend(
    handles=_legend_handles(runs),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=min(len(runs), 6),
    fontsize=7.5,
  )
  fig.tight_layout()
  return _save(fig, output)


def plot_loss_curves(
  runs: dict[str, pd.DataFrame],
  output: str | Path,
) -> Path:
  """2×3 grid: (train, val) × (regression, classification, auxiliary) losses."""
  _set_paper_style()
  fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=False)

  splits = ["train", "val"]
  roles = ["regression", "classification", "auxiliary"]

  for r, split in enumerate(splits):
    for c, role in enumerate(roles):
      ax = axes[r, c]
      for i, model in enumerate(_ordered_models(runs)):
        df = runs[model]
        col = _resolve_loss_column(df, split, role)
        if col is None:
          continue
        color = _color_for_model(model, fallback_index=i)
        x = df["epoch"].to_numpy()
        y = df[col].to_numpy(dtype=float)
        ax.plot(x, y, color=color, alpha=0.18, linewidth=0.9, zorder=2)
        if len(y) > 2:
          ax.plot(x, _ema(y), color=color, alpha=0.95, linewidth=1.4, zorder=3)
        else:
          ax.plot(x, y, color=color, alpha=0.95, linewidth=1.4, zorder=3)
      ax.set_title(f"{split} — {role}", fontsize=9.5)
      ax.set_xlabel("Epoch")
      ax.set_ylabel("Loss")
      ax.grid(True, alpha=0.25, linewidth=0.5)

  fig.suptitle(
    "Training and validation losses (regression unifies box/GIoU; aux unifies DFL/L1)",
    fontsize=10.5,
    y=1.0,
  )
  fig.legend(
    handles=_legend_handles(runs),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=min(len(runs), 6),
    fontsize=7.5,
  )
  fig.tight_layout()
  return _save(fig, output)


def _best_metric_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
  rows = []
  for model, df in runs.items():
    row: dict[str, float | str | int] = {"model": model, "family": _family_of(model)}
    for key, col in _METRIC_COLS.items():
      if col in df.columns:
        idx = int(df[col].idxmax())
        row[f"best_{key}"] = float(df[col].iloc[idx])
        row[f"best_{key}_epoch"] = int(df["epoch"].iloc[idx])
      else:
        row[f"best_{key}"] = float("nan")
        row[f"best_{key}_epoch"] = -1
    row["epochs_trained"] = int(df["epoch"].max())
    rows.append(row)
  return pd.DataFrame(rows)


def plot_best_metric_bars(
  runs: dict[str, pd.DataFrame],
  output: str | Path,
  *,
  metric: str = "map50_95",
) -> Path:
  """Horizontal lollipop of best validation metric per model; winner starred."""
  _set_paper_style()
  table = _best_metric_table(runs).sort_values(f"best_{metric}", ascending=True)

  colors = [_color_for_model(m) for m in table["model"]]
  fig, ax = plt.subplots(figsize=(6.5, max(3, 0.45 * len(table))))
  y_pos = np.arange(len(table))
  vals = table[f"best_{metric}"].to_numpy()
  epochs = table[f"best_{metric}_epoch"].to_numpy()
  winner_idx = int(np.argmax(vals))

  ax.hlines(y_pos, 0, vals, colors=colors, linewidth=1.5, zorder=2)
  ax.scatter(vals, y_pos, c=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)

  # Star the winner.
  ax.scatter(
    [vals[winner_idx]],
    [y_pos[winner_idx]],
    marker="*",
    s=260,
    c="#C9A227",
    edgecolors="#5A4A1F",
    linewidths=0.8,
    zorder=5,
  )

  xmax = max(float(vals.max()), 1e-3)
  for i, (v, ep) in enumerate(zip(vals, epochs, strict=True)):
    ax.text(
      v + xmax * 0.015,
      i,
      f"{v:.3f}  (ep {ep})",
      va="center",
      fontsize=7.5,
    )

  ax.set_yticks(list(y_pos))
  ax.set_yticklabels(table["model"])
  ax.set_xlabel(f"Best {metric.replace('_', '@').replace('map', 'mAP')}")
  ax.set_title(f"Best validation {metric} (★ = top)")
  ax.set_xlim(0, xmax * 1.20)
  sns.despine(ax=ax, left=True)
  ax.tick_params(axis="y", length=0)
  return _save(fig, output)


def plot_full_results_grid(
  runs: dict[str, pd.DataFrame],
  output: str | Path,
) -> Path:
  """2×5 grid mirroring Ultralytics ``results.png`` layout, overlayed by model.

  Row 1: train regression / cls / aux loss, precision, recall.
  Row 2: val regression / cls / aux loss, mAP@0.5, mAP@0.5:0.95.
  Loss roles unify YOLO (box/dfl) and RT-DETR (giou/l1) columns.
  """
  _set_paper_style()
  fig, axes = plt.subplots(2, 5, figsize=(15, 6.8), sharex=False)

  panels = [
    [
      ("loss", "train", "regression", "train regression loss"),
      ("loss", "train", "classification", "train cls loss"),
      ("loss", "train", "auxiliary", "train aux loss"),
      ("metric", _METRIC_COLS["precision"], None, "Precision"),
      ("metric", _METRIC_COLS["recall"], None, "Recall"),
    ],
    [
      ("loss", "val", "regression", "val regression loss"),
      ("loss", "val", "classification", "val cls loss"),
      ("loss", "val", "auxiliary", "val aux loss"),
      ("metric", _METRIC_COLS["map50"], None, "mAP@0.5"),
      ("metric", _METRIC_COLS["map50_95"], None, "mAP@0.5:0.95"),
    ],
  ]

  for r in range(2):
    for c in range(5):
      ax = axes[r, c]
      kind, a, b, title = panels[r][c]
      for i, model in enumerate(_ordered_models(runs)):
        df = runs[model]
        if kind == "loss":
          col = _resolve_loss_column(df, a, b)
        else:
          col = a if a in df.columns else None
        if col is None:
          continue
        color = _color_for_model(model, fallback_index=i)
        x = df["epoch"].to_numpy()
        y = df[col].to_numpy(dtype=float)
        ax.plot(x, y, color=color, alpha=0.18, linewidth=0.9, zorder=2)
        if len(y) > 2:
          ax.plot(x, _ema(y), color=color, alpha=0.95, linewidth=1.4, zorder=3)
        else:
          ax.plot(x, y, color=color, alpha=0.95, linewidth=1.4, zorder=3)
      ax.set_title(title, fontsize=9.5)
      ax.set_xlabel("Epoch")
      ax.grid(True, alpha=0.25, linewidth=0.5)

  fig.suptitle(
    "Detection training: losses and validation metrics across models",
    fontsize=11.5,
    y=1.0,
  )
  fig.legend(
    handles=_legend_handles(runs),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=min(len(runs), 10),
    fontsize=8,
  )
  fig.tight_layout()
  return _save(fig, output)


def plot_lr_schedule(runs: dict[str, pd.DataFrame], output: str | Path) -> Path:
  """Learning-rate schedule overlay (lr/pg0)."""
  _set_paper_style()
  fig, ax = plt.subplots(figsize=(7, 4))
  for i, model in enumerate(_ordered_models(runs)):
    df = runs[model]
    if "lr/pg0" not in df.columns:
      continue
    color = _color_for_model(model, fallback_index=i)
    ax.plot(df["epoch"], df["lr/pg0"], color=color, linewidth=1.4, label=model)
  ax.set_yscale("log")
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Learning rate (pg0, log)")
  ax.set_title("Learning rate schedule")
  ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
  ax.legend(handles=_legend_handles(runs), loc="best", fontsize=7, ncol=2)
  return _save(fig, output)


def render_all(
  experiments_dir: str | Path,
  output_dir: str | Path,
) -> dict[str, Path]:
  runs = load_runs(experiments_dir)
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)

  paths: dict[str, Path] = {}
  paths["full_grid"] = plot_full_results_grid(runs, out / "training_results_grid.png")
  paths["metrics"] = plot_metric_curves(runs, out / "metric_curves.png")
  paths["losses"] = plot_loss_curves(runs, out / "loss_curves.png")
  paths["best_map50_95"] = plot_best_metric_bars(
    runs, out / "best_map50_95_bars.png", metric="map50_95"
  )
  paths["best_map50"] = plot_best_metric_bars(
    runs, out / "best_map50_bars.png", metric="map50"
  )
  paths["lr"] = plot_lr_schedule(runs, out / "lr_schedule.png")

  # Also persist the best-epoch table for reference.
  table_path = out / "best_metrics.csv"
  _best_metric_table(runs).sort_values("best_map50_95", ascending=False).to_csv(
    table_path, index=False
  )
  paths["best_table"] = table_path
  return paths


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--experiments-dir",
    default="experiments/detection",
    help="Directory containing <model>/results.csv subfolders.",
  )
  parser.add_argument(
    "--output-dir",
    default="experiments/results/detection",
    help="Where to write the figures.",
  )
  args = parser.parse_args()

  written = render_all(args.experiments_dir, args.output_dir)
  for name, path in written.items():
    print(f"{name}: {path}")

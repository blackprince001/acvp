"""
Paper-quality plots for benchmark summaries.

All functions take a ``summary.json`` (list of BenchmarkResult dicts) and
write a single figure to disk. Built on seaborn for consistent styling.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# High-contrast qualitative palette — picked so adjacent lines in overlay
# plots stay distinguishable even when several models share a family.
FAMILY_COLORS: dict[str, str] = {
  "yolov5": "#1F77B4",  # blue
  "yolov8": "#2CA02C",  # green
  "yolov9": "#D62728",  # red
  "yolov10": "#9467BD",  # purple
  "yolov11": "#FF7F0E",  # orange
  "rtdetr": "#8C564B",  # brown
}
DEFAULT_COLOR = "#7F7F7F"  # neutral grey

# Per-model overrides for overlay plots. Within a family we vary hue/shade
# so individual sizes (n/s/m/l) remain distinguishable when plotted together.
MODEL_COLORS: dict[str, str] = {
  "yolov5n": "#1F77B4",  # blue
  "yolov5s": "#17BECF",  # cyan
  "yolov5m": "#0D4F8B",  # navy
  "yolov5l": "#74A9CF",  # steel blue
  "yolov8n": "#2CA02C",  # green
  "yolov8s": "#006D2C",  # dark green
  "yolov9c": "#D62728",  # red
  "yolov10m": "#9467BD",  # purple
  "yolov11m": "#FF7F0E",  # orange
  "rtdetr-l": "#8C564B",  # brown
}

# Fallback palette for unknown model names — high-contrast tab10-like.
_FALLBACK_PALETTE: tuple[str, ...] = (
  "#1F77B4",
  "#FF7F0E",
  "#2CA02C",
  "#D62728",
  "#9467BD",
  "#8C564B",
  "#E377C2",
  "#7F7F7F",
  "#BCBD22",
  "#17BECF",
)


def _family_of(name: str) -> str:
  m = re.match(r"(yolov?\d+|rtdetr)", name.lower())
  return m.group(1) if m else "other"


def _color_for(family: str) -> str:
  return FAMILY_COLORS.get(family, DEFAULT_COLOR)


def _color_for_model(name: str, *, fallback_index: int | None = None) -> str:
  """Return a per-model color; falls back to family color, then to a palette index."""
  if name in MODEL_COLORS:
    return MODEL_COLORS[name]
  family = _family_of(name)
  if family in FAMILY_COLORS:
    return FAMILY_COLORS[family]
  if fallback_index is not None:
    return _FALLBACK_PALETTE[fallback_index % len(_FALLBACK_PALETTE)]
  return DEFAULT_COLOR


def _set_paper_style() -> None:
  sns.set_theme(
    context="paper",
    style="whitegrid",
    rc={
      "figure.dpi": 150,
      "savefig.dpi": 300,
      "savefig.bbox": "tight",
      "savefig.pad_inches": 0.15,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "grid.alpha": 0.25,
      "grid.linewidth": 0.5,
      "legend.frameon": False,
      "legend.fontsize": 8,
      "font.family": "DejaVu Sans",
      "axes.titlesize": 11,
      "axes.titleweight": "medium",
      "axes.labelsize": 9.5,
      "xtick.labelsize": 8.5,
      "ytick.labelsize": 8.5,
      "lines.linewidth": 1.2,
    },
  )


def load_summary(path: str | Path) -> list[dict[str, Any]]:
  return json.loads(Path(path).read_text())


def _to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
  rows = []
  for r in results:
    rows.append(
      {
        "model": r["model_name"],
        "family": _family_of(r["model_name"]),
        "params_m": r["info"]["parameters"] / 1e6,
        "gflops": r["info"].get("gflops"),
        "weights_mb": r["info"]["weights_mb"],
        "map50": r["detection"]["map50"],
        "map50_95": r["detection"]["map50_95"],
        "precision": r["detection"]["precision"],
        "recall": r["detection"]["recall"],
        "f1": r["detection"]["f1"],
        "latency_mean_ms": r["latency"]["mean_ms"],
        "latency_p50_ms": r["latency"]["p50_ms"],
        "latency_p95_ms": r["latency"]["p95_ms"],
        "latency_p99_ms": r["latency"]["p99_ms"],
        "latency_min_ms": r["latency"]["min_ms"],
        "fps_mean": r["latency"]["fps_mean"],
      }
    )
  return pd.DataFrame(rows)


def _pareto_mask(
  xs: np.ndarray, ys: np.ndarray, *, lower_x_better: bool = False
) -> np.ndarray:
  """Return boolean mask of Pareto-optimal points (higher y is better)."""
  order = np.argsort(xs) if lower_x_better else np.argsort(-xs)
  mask = np.zeros(len(xs), dtype=bool)
  best_y = -np.inf
  for i in order:
    if ys[i] > best_y:
      mask[i] = True
      best_y = ys[i]
  return mask


def _draw_pareto(ax, xs, ys, mask, *, lower_x_better: bool = False):
  """Draw a stepped Pareto frontier line."""
  px, py = xs[mask], ys[mask]
  order = np.argsort(px) if lower_x_better else np.argsort(-px)
  px, py = px[order], py[order]
  ax.step(
    px,
    py,
    where="post",
    color="#5A5A5A",
    linewidth=1.0,
    linestyle="--",
    alpha=0.6,
    zorder=1,
    label="Pareto frontier",
  )


def _build_legend_handles(families: list[str]) -> list:
  """Build deduplicated legend handles for family colors."""
  from matplotlib.lines import Line2D

  seen = []
  handles = []
  for f in families:
    if f not in seen:
      seen.append(f)
      handles.append(
        Line2D(
          [0],
          [0],
          marker="o",
          color="w",
          markerfacecolor=_color_for(f),
          markersize=7,
          label=f,
          linewidth=0,
        )
      )
  return handles


def plot_metric_bars(
  results: list[dict[str, Any]],
  metric: str,
  output: str | Path,
  *,
  title: str | None = None,
  ylabel: str | None = None,
  sort: bool = True,
) -> Path:
  """Horizontal lollipop/dot plot of a single metric across models."""
  _set_paper_style()
  df = _to_dataframe(results)
  if sort:
    df = df.sort_values(metric, ascending=True)  # ascending for bottom-to-top

  colors = [_color_for(f) for f in df["family"]]
  fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(df))))
  y_pos = range(len(df))

  # stems
  ax.hlines(y_pos, 0, df[metric], colors=colors, linewidth=1.5, zorder=2)
  # dots
  ax.scatter(
    df[metric], y_pos, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.5
  )
  # value labels
  for i, (val, _) in enumerate(zip(df[metric], y_pos, strict=True)):
    ax.text(
      val + (df[metric].max() * 0.015), i, f"{val:.3f}", va="center", fontsize=7.5
    )

  ax.set_yticks(list(y_pos))
  ax.set_yticklabels(df["model"])
  ax.set_xlabel(ylabel or metric)
  ax.set_title(title or metric)
  ax.set_xlim(left=0, right=df[metric].max() * 1.12)
  ax.legend(
    handles=_build_legend_handles(df["family"].tolist()), loc="lower right", fontsize=7
  )
  sns.despine(ax=ax, left=True)
  ax.tick_params(axis="y", length=0)
  return _save(fig, output)


def plot_latency_box(results: list[dict[str, Any]], output: str | Path) -> Path:
  """Horizontal dumbbell/range plot: min → p50 → p95 → p99 per model."""
  _set_paper_style()
  df = _to_dataframe(results).sort_values("latency_mean_ms")

  fig, ax = plt.subplots(figsize=(6.5, max(3, 0.4 * len(df))))
  y_pos = range(len(df))

  for i, (_, row) in enumerate(df.iterrows()):
    c = _color_for(row["family"])
    # range line: min to p99
    ax.hlines(
      i, row["latency_min_ms"], row["latency_p99_ms"], colors=c, linewidth=1.5, zorder=2
    )
    # markers at each percentile
    ax.scatter(
      row["latency_min_ms"],
      i,
      c=c,
      s=25,
      marker="d",
      zorder=3,
      edgecolors="white",
      linewidths=0.4,
    )
    ax.scatter(
      row["latency_p50_ms"],
      i,
      c=c,
      s=55,
      marker="o",
      zorder=3,
      edgecolors="white",
      linewidths=0.5,
    )
    ax.scatter(
      row["latency_p95_ms"],
      i,
      c=c,
      s=40,
      marker="s",
      zorder=3,
      edgecolors="white",
      linewidths=0.4,
    )
    ax.scatter(
      row["latency_p99_ms"],
      i,
      c=c,
      s=30,
      marker="^",
      zorder=3,
      edgecolors="white",
      linewidths=0.4,
    )

  ax.set_yticks(list(y_pos))
  ax.set_yticklabels(df["model"])
  ax.set_xlabel("Latency (ms)")
  ax.set_title("Per-model inference latency")
  sns.despine(ax=ax, left=True)
  ax.tick_params(axis="y", length=0)

  # marker legend
  from matplotlib.lines import Line2D

  marker_handles = [
    Line2D(
      [0],
      [0],
      marker="d",
      color="w",
      markerfacecolor="#777",
      markersize=5,
      label="min",
      linewidth=0,
    ),
    Line2D(
      [0],
      [0],
      marker="o",
      color="w",
      markerfacecolor="#777",
      markersize=6,
      label="p50",
      linewidth=0,
    ),
    Line2D(
      [0],
      [0],
      marker="s",
      color="w",
      markerfacecolor="#777",
      markersize=5,
      label="p95",
      linewidth=0,
    ),
    Line2D(
      [0],
      [0],
      marker="^",
      color="w",
      markerfacecolor="#777",
      markersize=5,
      label="p99",
      linewidth=0,
    ),
  ]
  ax.legend(
    handles=marker_handles,
    loc="lower right",
    fontsize=7,
    title="Percentile",
    title_fontsize=7.5,
  )
  return _save(fig, output)


def plot_speed_accuracy(results: list[dict[str, Any]], output: str | Path) -> Path:
  """Scatter: FPS vs mAP@0.5:0.95 with Pareto frontier."""
  _set_paper_style()
  df = _to_dataframe(results)
  fig, ax = plt.subplots(figsize=(6.5, 4.5))

  colors = [_color_for(f) for f in df["family"]]
  sizes = 50 + 350 * (df["params_m"] - df["params_m"].min()) / max(
    df["params_m"].max() - df["params_m"].min(), 1e-9
  )

  ax.scatter(
    df["fps_mean"],
    df["map50_95"],
    c=colors,
    s=sizes,
    edgecolor="#444444",
    linewidth=0.4,
    alpha=0.85,
    zorder=3,
  )

  # Pareto frontier (higher FPS + higher mAP = better)
  xs, ys = df["fps_mean"].values, df["map50_95"].values
  mask = _pareto_mask(xs, ys)
  _draw_pareto(ax, xs, ys, mask)

  for _, row in df.iterrows():
    ax.annotate(
      row["model"],
      (row["fps_mean"], row["map50_95"]),
      fontsize=7,
      xytext=(5, 5),
      textcoords="offset points",
    )

  ax.set_xlabel("Throughput (FPS, higher →)")
  ax.set_ylabel("mAP@0.5:0.95 (higher →)")
  ax.set_title("Speed vs. accuracy")

  handles = _build_legend_handles(df["family"].tolist())
  from matplotlib.lines import Line2D

  handles.append(
    Line2D(
      [0], [0], linestyle="--", color="#5A5A5A", alpha=0.6, label="Pareto frontier"
    )
  )
  ax.legend(handles=handles, loc="best", fontsize=7, ncol=2)
  return _save(fig, output)


def plot_per_class_ap(
  results: list[dict[str, Any]],
  output: str | Path,
  *,
  metric_key: str = "per_class_ap50",
) -> Path:
  """Heatmap of per-class AP across models with muted earthy colormap."""
  _set_paper_style()
  rows = sorted(results, key=lambda r: r["model_name"])
  classes: list[str] = []
  for r in rows:
    for c in r["detection"].get(metric_key, {}):
      if c not in classes:
        classes.append(c)
  if not classes:
    raise RuntimeError("No per-class AP data available.")

  matrix = pd.DataFrame(
    [[r["detection"].get(metric_key, {}).get(c, 0.0) for c in classes] for r in rows],
    index=[r["model_name"] for r in rows],
    columns=classes,
  )

  # Muted sequential colormap: warm cream → terracotta → deep slate
  from matplotlib.colors import LinearSegmentedColormap

  cmap = LinearSegmentedColormap.from_list(
    "earthy_seq", ["#F5F0E8", "#C4A96A", "#C47E6E", "#6B5B6B"], N=256
  )

  fig, ax = plt.subplots(
    figsize=(max(4, 0.6 * len(classes) + 2), max(3, 0.35 * len(rows) + 0.5))
  )
  sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=0,
    vmax=1,
    cbar_kws={"label": "AP", "shrink": 0.8},
    linewidths=0.6,
    linecolor="white",
    ax=ax,
    annot_kws={"fontsize": 7.5},
  )
  ax.set_title(metric_key)
  ax.set_xlabel("Class")
  ax.set_ylabel("Model")
  ax.tick_params(axis="x", rotation=45)
  for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")
  return _save(fig, output)


def plot_size_vs_accuracy(results: list[dict[str, Any]], output: str | Path) -> Path:
  """Model size (params, M) vs mAP with Pareto frontier."""
  _set_paper_style()
  df = _to_dataframe(results)
  fig, ax = plt.subplots(figsize=(6.5, 4.5))

  colors = [_color_for(f) for f in df["family"]]
  ax.scatter(
    df["params_m"],
    df["map50_95"],
    c=colors,
    s=100,
    edgecolor="#444444",
    linewidth=0.4,
    alpha=0.85,
    zorder=3,
  )

  # Pareto frontier (lower params + higher mAP = better)
  xs, ys = df["params_m"].values, df["map50_95"].values
  mask = _pareto_mask(xs, ys, lower_x_better=True)
  _draw_pareto(ax, xs, ys, mask, lower_x_better=True)

  for _, row in df.iterrows():
    ax.annotate(
      row["model"],
      (row["params_m"], row["map50_95"]),
      fontsize=7,
      xytext=(5, 5),
      textcoords="offset points",
    )

  ax.set_xscale("log")
  ax.set_xlabel("Parameters (M, log scale, ← smaller)")
  ax.set_ylabel("mAP@0.5:0.95 (higher →)")
  ax.set_title("Model size vs. accuracy")

  handles = _build_legend_handles(df["family"].tolist())
  from matplotlib.lines import Line2D

  handles.append(
    Line2D(
      [0], [0], linestyle="--", color="#5A5A5A", alpha=0.6, label="Pareto frontier"
    )
  )
  ax.legend(handles=handles, loc="best", fontsize=7)
  return _save(fig, output)


def plot_metric_grid(results: list[dict[str, Any]], output: str | Path) -> Path:
  """Small-multiples lollipop grid: one panel per detection metric."""
  _set_paper_style()
  df = _to_dataframe(results).sort_values("map50_95", ascending=True)

  metrics = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("map50", "mAP@0.5"),
    ("map50_95", "mAP@0.5:0.95"),
  ]

  n = len(df)
  fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, max(4.5, 0.35 * n + 1.5)),
    sharey=True,
  )
  colors = [_color_for(f) for f in df["family"]]
  y_pos = np.arange(n)

  for ax, (key, label) in zip(axes.flat, metrics, strict=True):
    vals = df[key].to_numpy()
    ax.hlines(y_pos, 0, vals, colors=colors, linewidth=1.3, zorder=2)
    ax.scatter(
      vals, y_pos, c=colors, s=45, zorder=3, edgecolors="white", linewidths=0.5
    )
    xmax = max(float(vals.max()), 1e-3)
    for i, v in enumerate(vals):
      ax.text(v + xmax * 0.02, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, xmax * 1.18)
    ax.set_title(label, fontsize=10)
    ax.tick_params(axis="y", length=0)
    sns.despine(ax=ax, left=True)

  axes[0, 0].set_yticks(list(y_pos))
  axes[0, 0].set_yticklabels(df["model"])
  axes[1, 0].set_yticks(list(y_pos))
  axes[1, 0].set_yticklabels(df["model"])

  fig.suptitle("Detection metrics profile", fontsize=11, y=1.0)

  handles = _build_legend_handles(df["family"].tolist())
  fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    fontsize=7.5,
    ncol=min(len(handles), 6),
  )
  fig.tight_layout()
  return _save(fig, output)


def render_all(summary_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
  """Render the full set of paper figures from a summary.json."""
  results = load_summary(summary_path)
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)
  paths: dict[str, Path] = {}
  paths["map50"] = plot_metric_bars(
    results,
    "map50",
    out / "map50_bars.png",
    title="mAP@0.5",
    ylabel="mAP@0.5",
  )
  paths["map50_95"] = plot_metric_bars(
    results,
    "map50_95",
    out / "map50_95_bars.png",
    title="mAP@0.5:0.95",
    ylabel="mAP@0.5:0.95",
  )
  paths["fps"] = plot_metric_bars(
    results,
    "fps_mean",
    out / "fps_bars.png",
    title="Throughput",
    ylabel="FPS",
  )
  paths["latency_box"] = plot_latency_box(results, out / "latency_box.png")
  paths["speed_accuracy"] = plot_speed_accuracy(results, out / "speed_vs_accuracy.png")
  paths["size_accuracy"] = plot_size_vs_accuracy(results, out / "size_vs_accuracy.png")
  paths["metric_grid"] = plot_metric_grid(results, out / "metric_grid.png")
  try:
    paths["per_class_ap50"] = plot_per_class_ap(results, out / "per_class_ap50.png")
  except RuntimeError:
    pass
  return paths


def _save(fig, output: str | Path) -> Path:
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output, dpi=300)
  plt.close(fig)
  return output

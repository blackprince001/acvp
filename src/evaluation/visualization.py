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
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Family colour palette — keeps cross-figure visual identity consistent.
FAMILY_COLORS: dict[str, str] = {
  "yolov5": "#4C72B0",
  "yolov8": "#55A868",
  "yolov9": "#C44E52",
  "yolov10": "#8172B2",
  "yolo11": "#CCB974",
  "rtdetr": "#64B5CD",
}
DEFAULT_COLOR = "#888888"


def _family_of(name: str) -> str:
  m = re.match(r"(yolov?\d+|rtdetr)", name.lower())
  return m.group(1) if m else "other"


def _set_paper_style() -> None:
  sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="deep",
    rc={
      "figure.dpi": 130,
      "savefig.dpi": 200,
      "savefig.bbox": "tight",
      "axes.spines.top": False,
      "axes.spines.right": False,
      "grid.alpha": 0.3,
      "legend.frameon": False,
      "font.family": "DejaVu Sans",
      "axes.titlesize": 12,
      "axes.labelsize": 10,
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


def plot_metric_bars(
  results: list[dict[str, Any]],
  metric: str,
  output: str | Path,
  *,
  title: str | None = None,
  ylabel: str | None = None,
  sort: bool = True,
) -> Path:
  """Bar chart of a single metric across all models, coloured by family."""
  _set_paper_style()
  df = _to_dataframe(results)
  if sort:
    df = df.sort_values(metric, ascending=False)

  fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(df)), 4))
  sns.barplot(
    data=df,
    x="model",
    y=metric,
    hue="family",
    palette=FAMILY_COLORS,
    dodge=False,
    ax=ax,
  )
  for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
  ax.set_xlabel("")
  ax.set_ylabel(ylabel or metric)
  ax.set_title(title or metric)
  ax.tick_params(axis="x", rotation=45)
  for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")
  ax.legend(title="family", loc="best", fontsize=8)
  return _save(fig, output)


def plot_latency_box(results: list[dict[str, Any]], output: str | Path) -> Path:
  """Per-model latency distribution rendered from recorded percentiles."""
  _set_paper_style()
  df = _to_dataframe(results).sort_values("latency_mean_ms")
  stats = [
    {
      "label": row["model"],
      "med": row["latency_p50_ms"],
      "q1": row["latency_p50_ms"]
      - 0.5 * (row["latency_p95_ms"] - row["latency_p50_ms"]),
      "q3": row["latency_p95_ms"],
      "whislo": row["latency_min_ms"],
      "whishi": row["latency_p99_ms"],
      "fliers": [],
    }
    for _, row in df.iterrows()
  ]
  fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(df)), 4))
  ax.bxp(
    stats,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor="#cfd9ea", edgecolor="#36476b"),
    medianprops=dict(color="#36476b", linewidth=1.5),
  )
  ax.set_xticks(range(1, len(df) + 1))
  ax.set_xticklabels(df["model"].tolist(), rotation=45, ha="right")
  ax.set_ylabel("Latency (ms)")
  ax.set_title("Per-model inference latency")
  sns.despine(ax=ax)
  return _save(fig, output)


def plot_speed_accuracy(results: list[dict[str, Any]], output: str | Path) -> Path:
  """Scatter: FPS vs mAP@0.5:0.95 — Pareto frontier visible by eye."""
  _set_paper_style()
  df = _to_dataframe(results)
  fig, ax = plt.subplots(figsize=(6.5, 4.5))
  sns.scatterplot(
    data=df,
    x="fps_mean",
    y="map50_95",
    hue="family",
    size="params_m",
    palette=FAMILY_COLORS,
    sizes=(50, 400),
    edgecolor="black",
    linewidth=0.4,
    alpha=0.85,
    ax=ax,
  )
  for _, row in df.iterrows():
    ax.annotate(
      row["model"],
      (row["fps_mean"], row["map50_95"]),
      fontsize=8,
      xytext=(5, 5),
      textcoords="offset points",
    )
  ax.set_xlabel("Throughput (FPS, higher is better)")
  ax.set_ylabel("mAP@0.5:0.95 (higher is better)")
  ax.set_title("Speed vs. accuracy")
  ax.legend(loc="best", fontsize=8, ncol=2)
  return _save(fig, output)


def plot_per_class_ap(
  results: list[dict[str, Any]],
  output: str | Path,
  *,
  metric_key: str = "per_class_ap50",
) -> Path:
  """Heatmap of per-class AP across models."""
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
  fig, ax = plt.subplots(
    figsize=(max(4, 0.6 * len(classes) + 2), max(4, 0.35 * len(rows)))
  )
  sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "AP"},
    linewidths=0.5,
    linecolor="white",
    ax=ax,
  )
  ax.set_title(metric_key)
  ax.set_xlabel("class")
  ax.set_ylabel("model")
  return _save(fig, output)


def plot_size_vs_accuracy(results: list[dict[str, Any]], output: str | Path) -> Path:
  """Model size (params, M) vs mAP — useful for the discussion section."""
  _set_paper_style()
  df = _to_dataframe(results)
  fig, ax = plt.subplots(figsize=(6.5, 4.5))
  sns.scatterplot(
    data=df,
    x="params_m",
    y="map50_95",
    hue="family",
    palette=FAMILY_COLORS,
    s=120,
    edgecolor="black",
    linewidth=0.4,
    ax=ax,
  )
  for _, row in df.iterrows():
    ax.annotate(
      row["model"],
      (row["params_m"], row["map50_95"]),
      fontsize=8,
      xytext=(5, 5),
      textcoords="offset points",
    )
  ax.set_xscale("log")
  ax.set_xlabel("Parameters (M, log scale)")
  ax.set_ylabel("mAP@0.5:0.95")
  ax.set_title("Model size vs. accuracy")
  ax.legend(title="family", loc="best", fontsize=8)
  return _save(fig, output)


def plot_metric_grid(results: list[dict[str, Any]], output: str | Path) -> Path:
  """4-up grid: P, R, mAP@0.5, mAP@0.5:0.95 across models. Compact paper figure."""
  _set_paper_style()
  df = _to_dataframe(results).sort_values("map50_95", ascending=False)
  long = df.melt(
    id_vars=["model", "family"],
    value_vars=["precision", "recall", "map50", "map50_95"],
    var_name="metric",
    value_name="value",
  )
  g = sns.catplot(
    data=long,
    kind="bar",
    x="model",
    y="value",
    hue="family",
    col="metric",
    col_wrap=2,
    palette=FAMILY_COLORS,
    dodge=False,
    height=3.2,
    aspect=1.4,
    sharey=False,
  )
  g.set_xticklabels(rotation=45, horizontalalignment="right", fontsize=8)
  g.set_titles("{col_name}")
  g.set_axis_labels("", "score")
  for ax in g.axes.flat:
    for container in ax.containers:
      ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
  return _save(g.figure, output)


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
  fig.savefig(output)
  plt.close(fig)
  return output

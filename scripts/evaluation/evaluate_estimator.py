#!/usr/bin/env python3
"""Evaluate a trained density estimator on a held-out telemetry CSV.

Reports per-horizon and overall MSE / MAE / RMSE / R^2, in both normalized and
original feature space. Writes ``metrics.json`` and ``predictions.npz`` next
to the checkpoint (or to ``--output-dir`` if given).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import FEATURE_NAMES  # noqa: E402
from src.features.normalizer import FeatureNormalizer  # noqa: E402
from src.features.windowing import WindowBuilder  # noqa: E402
from src.models.ml_estimator.gru import GRUEstimator  # noqa: E402
from src.models.ml_estimator.lstm import LSTMEstimator  # noqa: E402
from src.models.ml_estimator.tcn import TCNEstimator  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402

REGISTRY = {"lstm": LSTMEstimator, "gru": GRUEstimator, "tcn": TCNEstimator}


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Evaluate an estimator checkpoint on a CSV.")
  p.add_argument("--csv", required=True, nargs="+", help="Held-out telemetry CSV(s).")
  p.add_argument("--model", choices=list(REGISTRY), required=True)
  p.add_argument("--checkpoint", required=True, help="Path to best.pt")
  p.add_argument("--normalizer", required=True, help="Path to normalizer.npz")
  p.add_argument("--input-len", type=int, default=10)
  p.add_argument("--horizon", type=int, default=5)
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--output-dir", default=None, help="Where to write metrics.json (default: alongside checkpoint).")
  p.add_argument("--device", default=None)
  return p.parse_args()


def load_features(csv_path: Path) -> np.ndarray:
  df = pd.read_csv(csv_path)
  return df[FEATURE_NAMES].to_numpy(dtype=np.float64)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
  err = y_pred - y_true
  mse = float(np.mean(err**2))
  mae = float(np.mean(np.abs(err)))
  rmse = float(np.sqrt(mse))
  ss_res = np.sum(err**2)
  ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
  r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
  return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def main() -> int:
  args = parse_args()
  setup_logger(log_level="INFO", log_file="experiments/logs/evaluate_estimator.log")
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

  out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
  out_dir.mkdir(parents=True, exist_ok=True)

  stats = np.load(args.normalizer)
  normalizer = FeatureNormalizer(method="standard")
  normalizer._center = stats["center"]
  normalizer._scale = stats["scale"]
  normalizer._fitted = True

  state = torch.load(args.checkpoint, map_location=device)
  cfg = state.get("config")
  if cfg is None:
    raise SystemExit("Checkpoint missing 'config' — re-train with the updated trainer.")
  model = REGISTRY[args.model](cfg).to(device)
  model.load_state_dict(state["model_state_dict"])
  model.eval()

  builder = WindowBuilder(input_len=args.input_len, horizon=args.horizon, stride=args.stride)
  min_len = args.input_len + args.horizon

  Xs, ys = [], []
  for csv in args.csv:
    seq = load_features(Path(csv))
    if len(seq) < min_len:
      logger.warning("Skipping short CSV {} ({} rows)", csv, len(seq))
      continue
    X, y = builder.build(normalizer.transform(seq))
    Xs.append(X)
    ys.append(y)
  if not Xs:
    raise SystemExit("No windows produced from eval CSVs.")
  X = np.concatenate(Xs, axis=0)
  y_true_n = np.concatenate(ys, axis=0)  # (N, horizon, F) normalized
  logger.info("Eval windows: {}", len(X))

  with torch.no_grad():
    preds = []
    bs = 256
    for i in range(0, len(X), bs):
      batch = torch.tensor(X[i : i + bs], dtype=torch.float32, device=device)
      out = model(batch).reshape(-1, args.horizon, cfg["output_size"]).cpu().numpy()
      preds.append(out)
  y_pred_n = np.concatenate(preds, axis=0)

  y_true = normalizer.inverse_transform(y_true_n.reshape(-1, y_true_n.shape[-1])).reshape(y_true_n.shape)
  y_pred = normalizer.inverse_transform(y_pred_n.reshape(-1, y_pred_n.shape[-1])).reshape(y_pred_n.shape)

  results: dict = {
    "n_windows": int(len(X)),
    "horizon": args.horizon,
    "input_len": args.input_len,
    "feature_names": FEATURE_NAMES,
    "overall_normalized": metrics_dict(y_true_n.reshape(-1), y_pred_n.reshape(-1)),
    "overall_original": metrics_dict(y_true.reshape(-1), y_pred.reshape(-1)),
    "per_horizon_step_original": [
      metrics_dict(y_true[:, h, :].reshape(-1), y_pred[:, h, :].reshape(-1))
      for h in range(args.horizon)
    ],
    "per_feature_original": {
      FEATURE_NAMES[f]: metrics_dict(y_true[:, :, f].reshape(-1), y_pred[:, :, f].reshape(-1))
      for f in range(y_true.shape[-1])
    },
  }

  metrics_path = out_dir / "metrics.json"
  with open(metrics_path, "w") as fh:
    json.dump(results, fh, indent=2)
  np.savez(out_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)

  o = results["overall_original"]
  logger.info(
    "{}: MSE={:.4f} MAE={:.4f} RMSE={:.4f} R2={:.4f} ({} windows)",
    args.model.upper(),
    o["mse"],
    o["mae"],
    o["rmse"],
    o["r2"],
    results["n_windows"],
  )
  logger.info("Metrics → {}", metrics_path)
  return 0


if __name__ == "__main__":
  sys.exit(main())

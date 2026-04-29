#!/usr/bin/env python3
"""Train a density estimator (LSTM/GRU/TCN) from a telemetry CSV.

The CSV is the one written by ``scripts/run_inference.py`` — columns
``frame_idx, timestamp, vehicle_count, occupancy_ratio, <FEATURE_NAMES...>``.
Only the FEATURE_NAMES columns are used as inputs/targets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

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
  p = argparse.ArgumentParser(
    description="Train a density estimator from a telemetry CSV."
  )
  p.add_argument(
    "--csv",
    required=True,
    nargs="+",
    help="One or more telemetry CSVs from run_inference. Windows are built per-CSV "
    "(no cross-file splicing) then concatenated.",
  )
  p.add_argument("--model", choices=list(REGISTRY), default="lstm")
  p.add_argument(
    "--output-dir", required=True, help="Where to write best.pt + normalizer.npz"
  )
  p.add_argument("--input-len", type=int, default=10)
  p.add_argument("--horizon", type=int, default=5)
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--val-split", type=float, default=0.2)
  p.add_argument("--epochs", type=int, default=200)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument(
    "--patience", type=int, default=30, help="Early-stop patience (epochs)."
  )
  p.add_argument("--hidden-size", type=int, default=128)
  p.add_argument("--num-layers", type=int, default=2)
  p.add_argument("--dropout", type=float, default=0.2)
  p.add_argument("--device", default=None)
  p.add_argument("--seed", type=int, default=42)
  return p.parse_args()


def load_features(csv_path: Path) -> np.ndarray:
  df = pd.read_csv(csv_path)
  missing = [c for c in FEATURE_NAMES if c not in df.columns]
  if missing:
    raise SystemExit(f"CSV is missing required feature columns: {missing}")
  arr = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
  logger.info(
    "Loaded {} rows × {} features from {}", arr.shape[0], arr.shape[1], csv_path
  )
  return arr


def main() -> int:
  args = parse_args()
  setup_logger(log_level="INFO", log_file="experiments/logs/train_estimator.log")
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

  per_csv_seqs = [load_features(Path(p)) for p in args.csv]
  n_feat = per_csv_seqs[0].shape[1]

  # Per-CSV time-ordered split (no cross-file splicing).
  train_seqs, val_seqs = [], []
  for seq in per_csv_seqs:
    split = int(len(seq) * (1.0 - args.val_split))
    if split > 0:
      train_seqs.append(seq[:split])
    if len(seq) - split > 0:
      val_seqs.append(seq[split:])

  normalizer = FeatureNormalizer(method="standard").fit(
    np.concatenate(train_seqs, axis=0)
  )
  np.savez(
    out_dir / "normalizer.npz", center=normalizer._center, scale=normalizer._scale
  )

  builder = WindowBuilder(
    input_len=args.input_len, horizon=args.horizon, stride=args.stride
  )
  min_len = args.input_len + args.horizon

  def windows_from(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for seq in seqs:
      if len(seq) < min_len:
        logger.warning("Skipping short segment of length {} (< {})", len(seq), min_len)
        continue
      X, y = builder.build(normalizer.transform(seq))
      Xs.append(X)
      ys.append(y)
    if not Xs:
      raise SystemExit("No windows produced — collected sequences too short.")
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

  X_tr, y_tr = windows_from(train_seqs)
  X_va, y_va = windows_from(val_seqs)
  logger.info("Train windows: {}  Val windows: {}", len(X_tr), len(X_va))

  X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
  y_tr_t = torch.tensor(y_tr.reshape(len(y_tr), -1), dtype=torch.float32)
  X_va_t = torch.tensor(X_va, dtype=torch.float32)
  y_va_t = torch.tensor(y_va.reshape(len(y_va), -1), dtype=torch.float32)

  train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True
  )
  val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=args.batch_size)

  cfg = {
    "input_size": n_feat,
    "output_size": n_feat,
    "horizon": args.horizon,
    "hidden_size": args.hidden_size,
    "num_layers": args.num_layers,
    "dropout": args.dropout,
  }
  model = REGISTRY[args.model](cfg).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=args.lr)
  sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
  loss_fn = nn.MSELoss()

  best_val = float("inf")
  best_path = out_dir / "best.pt"
  bad_epochs = 0

  for epoch in range(1, args.epochs + 1):
    model.train()
    tr_loss = 0.0
    for X, y in train_loader:
      X, y = X.to(device), y.to(device)
      opt.zero_grad()
      pred = model(X)
      loss = loss_fn(pred, y)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()
      tr_loss += loss.item() * X.size(0)
    tr_loss /= len(train_loader.dataset)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
      for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        va_loss += loss_fn(model(X), y).item() * X.size(0)
    va_loss /= max(len(val_loader.dataset), 1)
    sched.step(va_loss)

    logger.info("epoch {:>3} | train {:.5f} | val {:.5f}", epoch, tr_loss, va_loss)

    if va_loss < best_val:
      best_val = va_loss
      bad_epochs = 0
      torch.save({"model_state_dict": model.state_dict(), "config": cfg}, best_path)
    else:
      bad_epochs += 1
      if bad_epochs >= args.patience:
        logger.info("Early stop at epoch {} (best val {:.5f})", epoch, best_val)
        break

  logger.info("Best val MSE: {:.5f} → {}", best_val, best_path)
  return 0


if __name__ == "__main__":
  sys.exit(main())

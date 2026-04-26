#!/usr/bin/env python3
"""
Detection training entry point.

Prepares a YOLO-format dataset from ``data/segmented/`` (COCO annotations),
then trains the requested detection model variant(s).

Usage examples
--------------
Train a single model::

    uv run python scripts/03_train_detection.py --model yolov8n

Train multiple models sequentially::

    uv run python scripts/03_train_detection.py --model yolov8n yolov8s yolo11n

Override training parameters::

    uv run python scripts/03_train_detection.py \\
        --model yolov8n \\
        --epochs 50 \\
        --batch 8 \\
        --device cuda

Prepare data only (no training)::

    uv run python scripts/03_train_detection.py --prepare-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.models.detection.registry import get_model  # noqa: E402

DATA_SEGMENTED = PROJECT_ROOT / "data" / "segmented"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
DATA_YOLO = PROJECT_ROOT / "data" / "yolo_detection"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIG_DIR = PROJECT_ROOT / "configs" / "models" / "detection"

# COCO category id → YOLO class index mapping
# Our COCO uses id=1 (road) and id=2 (vehicles).
# We re-index to 0-based: road=0, vehicles=1
COCO_CAT_TO_YOLO: dict[int, int] = {1: 0, 2: 1}
CLASS_NAMES: list[str] = ["road", "vehicles"]

ALL_MODELS = [
  # YOLOv5
  "yolov5n",
  "yolov5s",
  "yolov5m",
  "yolov5l",
  "yolov5x",
  # YOLOv8
  "yolov8n",
  "yolov8s",
  "yolov8m",
  "yolov8l",
  "yolov8x",
  # YOLOv9
  "yolov9s",
  "yolov9m",
  "yolov9c",
  "yolov9e",
  # YOLOv10
  "yolov10n",
  "yolov10s",
  "yolov10m",
  "yolov10l",
  "yolov10x",
  "yolov10b",
  # YOLO11
  "yolo11n",
  "yolo11s",
  "yolo11m",
  "yolo11l",
  "yolo11x",
  # RT-DETR
  "rtdetr-l",
  "rtdetr-x",
]


def load_split(split_file: Path) -> set[str]:
  """Read a split manifest and return a set of filenames."""
  return {line.strip() for line in split_file.read_text().splitlines() if line.strip()}


def coco_bbox_to_yolo(
  bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
  """Convert COCO [x, y, w, h] to YOLO normalised [cx, cy, w, h].

  Args:
      bbox: COCO bounding box ``[x_min, y_min, width, height]``.
      img_w: Image width in pixels.
      img_h: Image height in pixels.

  Returns:
      Normalised ``(cx, cy, w, h)`` in range [0, 1].
  """
  x, y, w, h = bbox
  cx = (x + w / 2) / img_w
  cy = (y + h / 2) / img_h
  nw = w / img_w
  nh = h / img_h
  # Clamp to [0, 1] to handle any annotation edge cases
  cx = max(0.0, min(1.0, cx))
  cy = max(0.0, min(1.0, cy))
  nw = max(0.0, min(1.0, nw))
  nh = max(0.0, min(1.0, nh))
  return cx, cy, nw, nh


def prepare_yolo_dataset(
  force: bool = False,
  symlink_images: bool = True,
) -> Path:
  """Convert COCO annotations + split manifests → YOLO-format dataset.

  Creates the following layout under ``data/yolo_detection/``::

      data/yolo_detection/
        images/
          train/   <- symlinks (or copies) to data/segmented/*.jpg
          val/
          test/
        labels/
          train/   <- YOLO .txt label files
          val/
          test/
        data.yaml  <- Ultralytics dataset descriptor

  Args:
      force: Re-create the dataset even if ``data.yaml`` already exists.
      symlink_images: Use symlinks instead of copying images (faster, saves disk).

  Returns:
      Path to the generated ``data.yaml``.
  """
  data_yaml_path = DATA_YOLO / "data.yaml"
  if data_yaml_path.exists() and not force:
    logger.info(
      f"YOLO dataset already prepared at {DATA_YOLO}. Use --force to regenerate."
    )
    return data_yaml_path

  logger.info("Loading COCO annotations …")
  ann_path = DATA_SEGMENTED / "annotations.json"
  with open(ann_path) as f:
    coco = json.load(f)

  # Build lookup maps
  img_id_to_meta: dict[int, dict] = {img["id"]: img for img in coco["images"]}
  img_fname_to_id: dict[str, int] = {
    img["file_name"]: img["id"] for img in coco["images"]
  }

  # Group annotations by image_id
  anns_by_img: dict[int, list[dict]] = {}
  for ann in coco["annotations"]:
    iid = ann["image_id"]
    anns_by_img.setdefault(iid, []).append(ann)

  # Load split manifests
  splits = {
    "train": load_split(DATA_SPLITS / "train.txt"),
    "val": load_split(DATA_SPLITS / "val.txt"),
    "test": load_split(DATA_SPLITS / "test.txt"),
  }

  # Create directory layout
  for split in splits:
    (DATA_YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATA_YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)

  # Populate splits
  stats: dict[str, dict] = {}
  for split_name, filenames in splits.items():
    img_dir = DATA_YOLO / "images" / split_name
    lbl_dir = DATA_YOLO / "labels" / split_name
    n_imgs = 0
    n_anns = 0

    for fname in filenames:
      if fname not in img_fname_to_id:
        logger.warning(
          f"  Frame {fname!r} in split manifest not found in annotations — skipping."
        )
        continue

      img_id = img_fname_to_id[fname]
      meta = img_id_to_meta[img_id]
      img_w, img_h = meta["width"], meta["height"]

      # Link / copy image
      src_img = DATA_SEGMENTED / fname
      dst_img = img_dir / fname
      if not dst_img.exists():
        if symlink_images:
          dst_img.symlink_to(src_img.resolve())
        else:
          shutil.copy2(src_img, dst_img)

      # Write YOLO label file
      label_file = lbl_dir / (Path(fname).stem + ".txt")
      lines: list[str] = []
      for ann in anns_by_img.get(img_id, []):
        cat_id = ann["category_id"]
        yolo_cls = COCO_CAT_TO_YOLO.get(cat_id)
        if yolo_cls is None:
          continue
        cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
        lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        n_anns += 1

      label_file.write_text("\n".join(lines))
      n_imgs += 1

    stats[split_name] = {"images": n_imgs, "annotations": n_anns}
    logger.info(f"  {split_name:5s}: {n_imgs} images, {n_anns} annotations")

  # Write data.yaml
  data_yaml = {
    "path": str(DATA_YOLO.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
  }
  with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

  logger.success(f"YOLO dataset written to {DATA_YOLO}")
  logger.info(f"  Stats: {stats}")
  return data_yaml_path


def train_model(
  model_name: str,
  data_yaml: Path,
  device: str,
  overrides: dict,
) -> dict:
  """Instantiate and train a single detection model.

  Args:
      model_name: E.g. ``"yolov8n"``.
      data_yaml: Path to the YOLO dataset descriptor.
      device: Compute device.
      overrides: Training hyper-parameter overrides.

  Returns:
      Result dict from :meth:`BaseDetectionModel.train`.
  """
  output_dir = EXPERIMENTS_DIR / "detection" / model_name
  output_dir.mkdir(parents=True, exist_ok=True)

  logger.info(f"[{model_name}] Loading model …")
  model = get_model(model_name, device=device)

  logger.info(f"[{model_name}] Starting training → {output_dir}")
  result = model.train(
    data_yaml=data_yaml,
    output_dir=output_dir,
    **overrides,
  )
  logger.success(
    f"[{model_name}] Done. Best checkpoint: {result.get('best_checkpoint')}"
  )
  return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Prepare YOLO dataset and train detection model(s).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "--model",
    nargs="+",
    default=["yolov8n"],
    choices=ALL_MODELS,
    metavar="MODEL",
    help=f"Model(s) to train. Choices: {ALL_MODELS}. (default: yolov8n)",
  )
  parser.add_argument(
    "--device",
    default="cuda",
    help="Compute device: cuda, cpu, mps, 0, 1, … (default: cuda)",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override number of training epochs from the model config.",
  )
  parser.add_argument(
    "--batch",
    type=int,
    default=None,
    help="Override batch size from the model config.",
  )
  parser.add_argument(
    "--image-size",
    "--imgsz",
    type=int,
    default=None,
    dest="image_size",
    help="Override training image size (default: 640).",
  )
  parser.add_argument(
    "--prepare-only",
    action="store_true",
    help="Only prepare the YOLO dataset; do not train.",
  )
  parser.add_argument(
    "--force-prepare",
    action="store_true",
    help="Re-create the YOLO dataset even if it already exists.",
  )
  parser.add_argument(
    "--no-symlink",
    action="store_true",
    help="Copy images instead of symlinking (slower but standalone).",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)

  # Configure loguru
  logger.remove()
  logger.add(sys.stderr, level="INFO", colorize=True)
  log_file = EXPERIMENTS_DIR / "detection" / "train.log"
  log_file.parent.mkdir(parents=True, exist_ok=True)
  logger.add(log_file, level="DEBUG", rotation="10 MB")

  logger.info("===  Data Preparation ===")
  data_yaml = prepare_yolo_dataset(
    force=args.force_prepare,
    symlink_images=not args.no_symlink,
  )

  if args.prepare_only:
    logger.info("--prepare-only flag set; skipping training.")
    return

  logger.info("=== Training ===")

  # Build overrides dict (only include keys the caller explicitly set)
  overrides: dict = {}
  if args.epochs is not None:
    overrides["epochs"] = args.epochs
  if args.batch is not None:
    overrides["batch_size"] = args.batch
  if args.image_size is not None:
    overrides["image_size"] = args.image_size

  results_summary: dict[str, dict] = {}
  for model_name in args.model:
    try:
      result = train_model(
        model_name=model_name,
        data_yaml=data_yaml,
        device=args.device,
        overrides=overrides,
      )
      results_summary[model_name] = {
        "status": "success",
        "best_checkpoint": str(result.get("best_checkpoint", "")),
      }
    except Exception as exc:  # noqa: BLE001
      logger.error(f"[{model_name}] Training failed: {exc}")
      results_summary[model_name] = {"status": "failed", "error": str(exc)}

  # Summary
  logger.info("=== Training Summary ===")
  for name, info in results_summary.items():
    status = info["status"]
    if status == "success":
      logger.success(f"  ✓ {name}: {info['best_checkpoint']}")
    else:
      logger.error(f"  ✗ {name}: {info.get('error', 'unknown error')}")


if __name__ == "__main__":
  main()

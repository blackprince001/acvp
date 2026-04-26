"""Train/val/test split management for traffic scenes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

VALID_SCENE_TYPES = {"highway", "urban", "suburban", "intersection"}


@dataclass
class SplitConfig:
  """Configuration for scene-level splits."""

  train_ratio: float = 0.7
  val_ratio: float = 0.15
  test_ratio: float = 0.15
  seed: int = 42
  stratify: bool = True


class SceneSplitManager:
  """Manage scene-level train/val/test splits.

  Ensures:
  - Scene-level split (not frame-level) to prevent data leakage
  - Stratified by scene type (highway, urban, etc.)
  - Reproducible with seed
  """

  def __init__(self, config: SplitConfig | None = None) -> None:
    self.config = config or SplitConfig()
    self._validate_config()

    random.seed(self.config.seed)

    self.train_scenes: list[str] = []
    self.val_scenes: list[str] = []
    self.test_scenes: list[str] = []

  def _validate_config(self) -> None:
    total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
    if not (0.99 <= total <= 1.01):
      raise ValueError(f"Split ratios must sum to 1.0, got {total}")

  def create_splits(
    self,
    scene_dirs: list[Path] | list[str],
    output_dir: Path | str | None = None,
  ) -> dict[str, list[str]]:
    """Create scene-level splits from scene directories.

    Args:
        scene_dirs: List of paths to scene directories.
        output_dir: If set, write split manifests to this directory.

    Returns:
        Dict with keys 'train', 'val', 'test', each a list of scene names.
    """
    scene_dirs = [Path(p) for p in scene_dirs]

    scene_info = self._collect_scene_info(scene_dirs)

    if self.config.stratify:
      splits = self._split_stratified(scene_info)
    else:
      splits = self._split_random(scene_info)

    self.train_scenes = splits["train"]
    self.val_scenes = splits["val"]
    self.test_scenes = splits["test"]

    logger.info(
      "Split: {} train, {} val, {} test",
      len(self.train_scenes),
      len(self.val_scenes),
      len(self.test_scenes),
    )

    if output_dir:
      self._save_manifests(output_dir)

    return splits

  def _collect_scene_info(self, scene_dirs: list[Path]) -> dict[str, dict]:
    info = {}
    for scene_dir in scene_dirs:
      name = scene_dir.name
      scene_type = "unknown"

      meta_path = scene_dir / "metadata.yaml"
      if meta_path.exists():
        try:
          with open(meta_path) as fh:
            data = yaml.safe_load(fh) or {}
          scene_type = data.get("scene", {}).get("type", "unknown")
        except Exception:
          pass

      info[name] = {"type": scene_type, "path": str(scene_dir)}

    return info

  def _split_stratified(self, scene_info: dict[str, dict]) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = {}
    for name, info in scene_info.items():
      scene_type = info["type"]
      if scene_type not in by_type:
        by_type[scene_type] = []
      by_type[scene_type].append(name)

    train, val, test = [], [], []

    for _, scenes in by_type.items():
      random.shuffle(scenes)

      n = len(scenes)
      n_train = int(n * self.config.train_ratio)
      n_val = int(n * self.config.val_ratio)

      train.extend(scenes[:n_train])
      val.extend(scenes[n_train : n_train + n_val])
      test.extend(scenes[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return {"train": train, "val": val, "test": test}

  def _split_random(self, scene_info: dict[str, dict]) -> dict[str, list[str]]:
    scenes = list(scene_info.keys())
    random.shuffle(scenes)

    n = len(scenes)
    n_train = int(n * self.config.train_ratio)
    n_val = int(n * self.config.val_ratio)

    return {
      "train": scenes[:n_train],
      "val": scenes[n_train : n_train + n_val],
      "test": scenes[n_train + n_val :],
    }

  def _save_manifests(self, output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
      "train.txt": self.train_scenes,
      "val.txt": self.val_scenes,
      "test.txt": self.test_scenes,
    }

    for filename, scenes in manifests.items():
      manifest_path = output_dir / filename
      with open(manifest_path, "w") as fh:
        fh.write("\n".join(scenes))

    split_config = {
      "train_ratio": self.config.train_ratio,
      "val_ratio": self.config.val_ratio,
      "test_ratio": self.config.test_ratio,
      "seed": self.config.seed,
      "stratify": self.config.stratify,
      "train_count": len(self.train_scenes),
      "val_count": len(self.val_scenes),
      "test_count": len(self.test_scenes),
    }
    with open(output_dir / "split_config.yaml", "w") as fh:
      yaml.dump(split_config, fh)

    logger.info("Saved split manifests to {}", output_dir)


def load_splits(splits_dir: Path | str) -> dict[str, list[str]]:
  """Load split manifests from directory.

  Args:
      splits_dir: Path to directory with train.txt, val.txt, test.txt.

  Returns:
      Dict with keys 'train', 'val', 'test'.
  """
  splits_dir = Path(splits_dir)

  result = {}
  for split_name in ["train", "val", "test"]:
    manifest_path = splits_dir / f"{split_name}.txt"
    if manifest_path.exists():
      with open(manifest_path) as fh:
        result[split_name] = [line.strip() for line in fh if line.strip()]
    else:
      result[split_name] = []

  return result


def load_split_config(splits_dir: Path | str) -> dict:
  """Load split configuration."""
  splits_dir = Path(splits_dir)
  config_path = splits_dir / "split_config.yaml"

  if not config_path.exists():
    return {}

  with open(config_path) as fh:
    return yaml.safe_load(fh)


def get_split_scenes(
  splits_dir: Path | str,
  split: str = "train",
) -> list[str]:
  """Get list of scene names for a specific split.

  Args:
      splits_dir: Path to splits directory.
      split: One of 'train', 'val', 'test'.

  Returns:
      List of scene names.
  """
  splits = load_splits(splits_dir)
  return splits.get(split, [])

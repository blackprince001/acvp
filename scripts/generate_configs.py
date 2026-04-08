import os

import yaml

configs = {}

# BASE CONF
configs["configs/base.yaml"] = {
  "mode": "detect",
  "target_fps": 30,
  "density_window_size": 30,
  "project_dir": ".",
  "data_dir": "data",
  "experiment_dir": "experiments",
  "device": "cuda",
  "logging": {"level": "INFO"},
}

# DETECTION
yolo_det_models = [
  "yolov8n",
  "yolov8s",
  "yolov8m",
  "yolov8l",
  "yolov11n",
  "yolov11s",
  "yolov11m",
  "yolov11l",
]
for m in yolo_det_models:
  configs[f"configs/models/detection/{m}.yaml"] = {
    "model": {
      "name": m,
      "task": "detect",
      "pretrained": True,
      "pretrained_weights": f"{m}.pt",
    },
    "training": {
      "epochs": 100,
      "batch_size": 16,
      "image_size": 640,
      "optimizer": "AdamW",
      "lr": 0.001,
      "lr_scheduler": "cosine",
      "warmup_epochs": 5,
      "weight_decay": 0.0005,
      "patience": 20,
      "augmentations": {
        "brightness": 0.3,
        "contrast": 0.3,
        "blur": 0.1,
        "rotation": 5,
        "flip_lr": 0.5,
      },
      "classes": ["road", "vehicle"],
    },
  }

# SEGMENTATION
yolo_seg_models = ["yolov8n-seg", "yolov8s-seg", "yolov11n-seg", "yolov11s-seg"]
for m in yolo_seg_models:
  configs[f"configs/models/segmentation/{m}.yaml"] = {
    "model": {
      "name": m.split("-")[0],
      "task": "segment",
      "pretrained": True,
      "pretrained_weights": f"{m}.pt",
    },
    "training": {
      "epochs": 100,
      "batch_size": 16,
      "image_size": 640,
      "optimizer": "AdamW",
      "lr": 0.001,
      "lr_scheduler": "cosine",
      "warmup_epochs": 5,
      "weight_decay": 0.0005,
      "patience": 20,
      "augmentations": {
        "brightness": 0.3,
        "contrast": 0.3,
        "blur": 0.1,
        "rotation": 5,
        "flip_lr": 0.5,
      },
      "classes": ["road", "vehicle"],
    },
  }

# ML ESTIMATOR
configs["configs/models/ml_estimator/lstm.yaml"] = {
  "model": {
    "name": "lstm",
    "input_size": 10,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": False,
  },
  "training": {
    "epochs": 200,
    "batch_size": 64,
    "optimizer": "Adam",
    "lr": 0.001,
    "lr_scheduler": "reduce_on_plateau",
    "patience": 30,
    "loss": "mse",
    "clip_grad_norm": 1.0,
  },
}

configs["configs/models/ml_estimator/gru.yaml"] = {
  "model": {
    "name": "gru",
    "input_size": 10,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": False,
  },
  "training": {
    "epochs": 200,
    "batch_size": 64,
    "optimizer": "Adam",
    "lr": 0.001,
    "lr_scheduler": "reduce_on_plateau",
    "patience": 30,
    "loss": "mse",
    "clip_grad_norm": 1.0,
  },
}

configs["configs/models/ml_estimator/tcn.yaml"] = {
  "model": {
    "name": "tcn",
    "input_size": 10,
    "num_channels": [64, 64, 64, 64],
    "kernel_size": 3,
    "dropout": 0.2,
    "dilation_base": 2,
  },
  "training": {
    "epochs": 200,
    "batch_size": 64,
    "optimizer": "Adam",
    "lr": 0.001,
    "lr_scheduler": "reduce_on_plateau",
    "patience": 30,
    "loss": "mse",
    "clip_grad_norm": 1.0,
  },
}

# TRACKER
configs["configs/tracking/botsort.yaml"] = {
  "tracker_type": "botsort",
  "track_high_thresh": 0.5,
  "track_low_thresh": 0.1,
  "new_track_thresh": 0.6,
  "track_buffer": 30,
  "match_thresh": 0.8,
  "fuse_score": True,
}
configs["configs/tracking/bytetrack.yaml"] = {
  "tracker_type": "bytetrack",
  "track_high_thresh": 0.5,
  "track_low_thresh": 0.1,
  "new_track_thresh": 0.6,
  "track_buffer": 30,
  "match_thresh": 0.8,
  "fuse_score": True,
}

# EXPERIMENTS
configs["configs/experiments/baseline_detection.yaml"] = {
  "mode": "detect",
  "model": {"name": "yolov8n"},
  "tracking": {"tracker_type": "botsort"},
}
configs["configs/experiments/baseline_segmentation.yaml"] = {
  "mode": "segment",
  "model": {"name": "yolov8n-seg"},
  "tracking": {"tracker_type": "botsort"},
}
for name in ["lstm", "gru", "tcn"]:
  configs[f"configs/experiments/predictive_{name}.yaml"] = {
    "mode": "predict",
    "model": {"name": "yolov8s-seg"},
    "estimator": {"name": name},
    "tracking": {"tracker_type": "botsort"},
  }
configs["configs/experiments/edge_deployment.yaml"] = {
  "mode": "detect",
  "model": {"name": "yolov11n"},
  "target_fps": 30,
}

for path, data in configs.items():
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    yaml.dump(data, f, sort_keys=False, default_flow_style=False)

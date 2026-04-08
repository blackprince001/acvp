"""
Batch segmentation using Roboflow SAM-3 workflow.

This script runs inference using the Roboflow hosted API with the SAM-3 workflow
to segment road and vehicles in traffic images.
"""

import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from inference_sdk import InferenceHTTPClient
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WORKSPACE_NAME = "appiah"
WORKFLOW_ID = "sam-3"
CLASS_NAMES = ["road", "vehicles"]
INPUT_DIR = Path("data/roboflow_upload")
OUTPUT_DIR = Path("data/segmented")
MAX_WORKERS = 4
BATCH_SIZE = 50


def run_segmentation(image_path: Path, client: InferenceHTTPClient) -> dict | None:
  """Run segmentation on a single image."""
  try:
    result = client.run_workflow(
      workspace_name=WORKSPACE_NAME,
      workflow_id=WORKFLOW_ID,
      images={"image": str(image_path)},
      parameters={"className": CLASS_NAMES},
      use_cache=False,
    )
    return {
      "image_path": str(image_path),
      "image_name": image_path.name,
      "result": result,
    }
  except Exception as e:
    logger.error(f"Error processing {image_path.name}: {e}")
    return None


def save_mask_image(result: dict, output_dir: Path) -> str | None:
  """Save the mask visualization image."""
  if not result or not result[0].get("model_predictions"):
    return None

  pred = result[0]["model_predictions"]
  mask_b64 = pred.get("mask_visualization_image", {}).get("value")

  if not mask_b64:
    return None

  # Decode base64 and save as PNG
  image_data = base64.b64decode(mask_b64)
  image = Image.open(BytesIO(image_data))

  # Use same filename but .png extension
  output_path = output_dir / result["image_name"].replace(".jpg", ".png")
  image.save(output_path)
  return str(output_path)


def save_annotations_summary(results: list[dict], output_path: Path):
  """Save minimal annotations summary."""
  annotations = {
    "info": {
      "description": "Traffic segmentation dataset - SAM-3 workflow",
      "version": "1.0",
      "year": 2026,
      "contributor": "AAI Computer Vision Project",
      "date_created": "2026-04-08",
    },
    "categories": [
      {"id": 1, "name": "road", "supercategory": "terrain"},
      {"id": 2, "name": "vehicles", "supercategory": "vehicle"},
    ],
    "images": [],
    "annotations": [],
  }

  for res in results:
    if res is None or res.get("result") is None:
      continue

    result = res["result"]
    if not result or not result[0].get("model_predictions"):
      continue

    pred = result[0]["model_predictions"]
    image_info = pred.get("image", {})
    image_id = len(annotations["images"]) + 1

    image_name = res["image_name"].replace(".jpg", ".png")
    annotations["images"].append(
      {
        "id": image_id,
        "file_name": image_name,
        "width": image_info.get("width", 640),
        "height": image_info.get("height", 640),
      }
    )

    predictions = pred.get("predictions", [])
    for p in predictions:
      class_id = p.get("class_id", 0) + 1
      # Skip polygon points - just use bbox for minimal file size
      w = p.get("width", 0)
      h = p.get("height", 0)
      area = w * h

      annotations["annotations"].append(
        {
          "id": len(annotations["annotations"]) + 1,
          "image_id": image_id,
          "category_id": class_id,
          "bbox": [p.get("x", 0), p.get("y", 0), w, h],
          "area": area,
          "iscrowd": 0,
        }
      )

  with open(output_path, "w") as f:
    json.dump(annotations, f, indent=2)

  logger.info(f"Saved {len(annotations['annotations'])} annotations to {output_path}")


def process_batch(
  image_paths: list[Path], client: InferenceHTTPClient, output_dir: Path
) -> list[dict]:
  """Process a batch of images in parallel."""
  results = []
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
      executor.submit(run_segmentation, img, client): img for img in image_paths
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="Segmenting"):
      result = future.result()
      if result:
        save_mask_image(result["result"], output_dir)
      results.append(result)

  return results


def main():
  """Main entry point."""
  api_key = os.environ.get("ROBOFLOW_API_KEY")
  if not api_key:
    logger.error("ROBOFLOW_API_KEY environment variable not set")
    return 1

  client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com", api_key=api_key
  )

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  image_paths = sorted(INPUT_DIR.glob("*.jpg"))
  logger.info(f"Found {len(image_paths)} images to process")

  all_results = []
  for i in range(0, len(image_paths), BATCH_SIZE):
    batch = image_paths[i : i + BATCH_SIZE]
    logger.info(
      f"Processing batch {i // BATCH_SIZE + 1}/{(len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE}"
    )
    results = process_batch(batch, client, OUTPUT_DIR)
    all_results.extend(results)

  # Save minimal annotations (bbox only, no polygons)
  save_annotations_summary(all_results, OUTPUT_DIR / "annotations.json")

  logger.info(
    f"Segmentation complete! Saved {len(list(OUTPUT_DIR.glob('*.png')))} mask images"
  )
  return 0


if __name__ == "__main__":
  exit(main())

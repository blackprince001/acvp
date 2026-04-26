"""Metadata schema validation for scene metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

VALID_SCENE_TYPES = {"highway", "urban", "suburban", "intersection"}
VALID_WEATHER = {"clear", "rain", "fog", "night", "cloudy", "snow"}


@dataclass
class ValidationError:
  """Single validation error."""

  field: str
  message: str
  value: any = field(default=None)


@dataclass
class ValidationResult:
  """Result of metadata validation."""

  valid: bool
  errors: list[ValidationError] = field(default_factory=list)

  def __str__(self) -> str:
    if self.valid:
      return "Valid"
    lines = [f"Invalid ({len(self.errors)} error(s)):"]
    for err in self.errors:
      lines.append(f"  - {err.field}: {err.message}")
    return "\n".join(lines)


class SceneMetadataValidator:
  """Validate scene metadata against the schema.

  Schema (from SPEC Section 3.1):

  scene:
    name: string
    type: highway | urban | suburban | intersection
    location: string
    camera:
      height_m: float (>= 0)
      angle_deg: float (0-90)
      resolution: [int, int]
      fps: float (> 0)
    recording:
      date: string (YYYY-MM-DD)
      weather: clear | rain | fog | night | cloudy | snow
      duration_seconds: float (>= 0)
    road:
      lanes: int (>= 0)
      has_sidewalk: bool
      has_service_road: bool
  """

  def __init__(self) -> None:
    self.errors: list[ValidationError] = []

  def validate(self, data: dict) -> ValidationResult:
    """Validate metadata dictionary.

    Args:
        data: Raw metadata dictionary loaded from YAML.

    Returns:
        ValidationResult with valid flag and list of errors.
    """
    self.errors = []

    if not isinstance(data, dict):
      return ValidationResult(False, [ValidationError("", "Root must be a dictionary")])

    scene = data.get("scene")
    if scene is None:
      return ValidationResult(
        False, [ValidationError("scene", "Missing required 'scene' key")]
      )

    self._validate_scene(scene)
    return ValidationResult(len(self.errors) == 0, self.errors)

  def _validate_scene(self, scene: dict) -> None:
    name = scene.get("name")
    if not name:
      self.errors.append(ValidationError("scene.name", "Missing required field"))
    elif not isinstance(name, str):
      self.errors.append(ValidationError("scene.name", "Must be a string", name))

    scene_type = scene.get("type")
    if scene_type is None:
      self.errors.append(ValidationError("scene.type", "Missing required field"))
    elif scene_type not in VALID_SCENE_TYPES:
      self.errors.append(
        ValidationError(
          "scene.type",
          f"Must be one of {VALID_SCENE_TYPES}",
          scene_type,
        )
      )

    location = scene.get("location")
    if location and not isinstance(location, str):
      self.errors.append(
        ValidationError("scene.location", "Must be a string", location)
      )

    self._validate_camera(scene.get("camera", {}))
    self._validate_recording(scene.get("recording", {}))
    self._validate_road(scene.get("road", {}))

  def _validate_camera(self, camera: dict) -> None:
    height = camera.get("height_m")
    if height is None:
      self.errors.append(
        ValidationError("scene.camera.height_m", "Missing required field")
      )
    elif not isinstance(height, (int, float)) or height < 0:
      self.errors.append(
        ValidationError(
          "scene.camera.height_m", "Must be a non-negative number", height
        )
      )

    angle = camera.get("angle_deg")
    if angle is None:
      self.errors.append(
        ValidationError("scene.camera.angle_deg", "Missing required field")
      )
    elif not isinstance(angle, (int, float)) or not (0 <= angle <= 90):
      self.errors.append(
        ValidationError("scene.camera.angle_deg", "Must be between 0 and 90", angle)
      )

    res = camera.get("resolution")
    if res is None:
      self.errors.append(
        ValidationError("scene.camera.resolution", "Missing required field")
      )
    elif not isinstance(res, (list, tuple)) or len(res) != 2:
      self.errors.append(
        ValidationError("scene.camera.resolution", "Must be [width, height]", res)
      )
    elif not all(isinstance(x, int) and x > 0 for x in res):
      self.errors.append(
        ValidationError(
          "scene.camera.resolution", "Values must be positive integers", res
        )
      )

    fps = camera.get("fps")
    if fps is None:
      self.errors.append(ValidationError("scene.camera.fps", "Missing required field"))
    elif not isinstance(fps, (int, float)) or fps <= 0:
      self.errors.append(
        ValidationError("scene.camera.fps", "Must be a positive number", fps)
      )

  def _validate_recording(self, recording: dict) -> None:
    date = recording.get("date")
    if date and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date)):
      self.errors.append(
        ValidationError("scene.recording.date", "Must be YYYY-MM-DD format", date)
      )

    weather = recording.get("weather")
    if weather and weather not in VALID_WEATHER:
      self.errors.append(
        ValidationError(
          "scene.recording.weather",
          f"Must be one of {VALID_WEATHER}",
          weather,
        )
      )

    duration = recording.get("duration_seconds")
    if duration is not None and (
      not isinstance(duration, (int, float)) or duration < 0
    ):
      self.errors.append(
        ValidationError(
          "scene.recording.duration_seconds",
          "Must be a non-negative number",
          duration,
        )
      )

  def _validate_road(self, road: dict) -> None:
    lanes = road.get("lanes")
    if lanes is not None and (not isinstance(lanes, int) or lanes < 0):
      self.errors.append(
        ValidationError("scene.road.lanes", "Must be a non-negative integer", lanes)
      )

    for field_name in ["has_sidewalk", "has_service_road"]:
      val = road.get(field_name)
      if val is not None and not isinstance(val, bool):
        self.errors.append(
          ValidationError(f"scene.road.{field_name}", "Must be a boolean", val)
        )


def load_and_validate(meta_path: str | Path) -> tuple[dict, ValidationResult]:
  """Load metadata from YAML and validate it.

  Args:
      meta_path: Path to metadata.yaml file.

  Returns:
      (raw_data, validation_result)
  """
  meta_path = Path(meta_path)

  if not meta_path.exists():
    return {}, ValidationResult(
      False, [ValidationError("", f"File not found: {meta_path}")]
    )

  with open(meta_path) as fh:
    data = yaml.safe_load(fh) or {}

  validator = SceneMetadataValidator()
  result = validator.validate(data)
  return data, result


def create_scene_metadata(
  name: str,
  scene_type: str,
  location: str = "Unknown",
  camera_height_m: float = 8.0,
  camera_angle_deg: float = 30,
  resolution: tuple[int, int] = (1920, 1080),
  fps: float = 30.0,
  weather: str = "clear",
  duration_seconds: float = 0.0,
  road_lanes: int = 3,
  has_sidewalk: bool = False,
  has_service_road: bool = False,
  output_path: str | Path | None = None,
) -> dict:
  """Create a valid scene metadata dictionary.

  Args:
      name: Scene name.
      scene_type: One of highway, urban, suburban, intersection.
      location: Location string.
      camera_height_m: Camera height in meters.
      camera_angle_deg: Camera angle in degrees.
      resolution: (width, height) resolution.
      fps: Frames per second.
      weather: Weather condition.
      duration_seconds: Recording duration.
      road_lanes: Number of lanes.
      has_sidewalk: Whether sidewalk exists.
      has_service_road: Whether service road exists.
      output_path: If set, write to this path.

  Returns:
      Metadata dictionary.
  """
  metadata = {
    "scene": {
      "name": name,
      "type": scene_type,
      "location": location,
      "camera": {
        "height_m": camera_height_m,
        "angle_deg": camera_angle_deg,
        "resolution": list(resolution),
        "fps": fps,
      },
      "recording": {
        "weather": weather,
        "duration_seconds": duration_seconds,
      },
      "road": {
        "lanes": road_lanes,
        "has_sidewalk": has_sidewalk,
        "has_service_road": has_service_road,
      },
    }
  }

  if output_path:
    with open(output_path, "w") as fh:
      yaml.dump(metadata, fh, default_flow_style=False)
    logger.info("Wrote metadata to {}", output_path)

  return metadata

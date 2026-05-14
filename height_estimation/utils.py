from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from height_estimation.detector import detect_buildings
from height_estimation.geometry import compute_shadow_length, estimate_building_height
from height_estimation.shadow_analysis import associate_nearest_shadow, detect_shadows
from height_estimation.visualization import draw_height_overlay, matplotlib_preview


@dataclass
class HeightEstimationConfig:
    meters_per_pixel: float = 0.2
    solar_elevation_angle: float = 45.0
    min_shadow_area: float = 120.0
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    debug_matplotlib: bool = False


def _decode_data_url_to_bgr(data_url: str) -> Any | None:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if not data_url or "," not in data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[1]
        binary = base64.b64decode(encoded)
        arr = np.frombuffer(binary, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image_bgr
    except Exception:
        return None


def _encode_bgr_to_data_url(image_bgr: Any, ext: str = ".jpg") -> str:
    import cv2  # type: ignore

    success, encoded = cv2.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError("Failed to encode overlay image.")
    mime = "image/jpeg" if ext.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(encoded.tobytes()).decode('utf-8')}"


def detect_shadows_for_buildings(
    source_image_bgr: Any,
    detected_assets: List[Dict[str, Any]],
    config: HeightEstimationConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any]:
    """Core shadow-based height estimation logic on top of existing building detections."""
    buildings = detect_buildings(detected_assets, source_image_bgr.shape)
    shadow_mask, shadow_contours = detect_shadows(
        source_image_bgr,
        min_shadow_area=config.min_shadow_area,
        adaptive_block_size=config.adaptive_block_size,
        adaptive_c=config.adaptive_c,
    )

    building_results: List[Dict[str, Any]] = []
    for building in buildings:
        bbox = building["bbox"]
        matched_shadow = associate_nearest_shadow(bbox, shadow_contours)
        shadow_info = compute_shadow_length(bbox, matched_shadow)

        if shadow_info is None:
            shadow_length_pixels = 0.0
            shadow_length_meters = 0.0
            height_meters = 0.0
            start_point = None
            end_point = None
        else:
            shadow_length_pixels = float(shadow_info["shadow_length_pixels"])
            shadow_length_meters = shadow_length_pixels * float(config.meters_per_pixel)
            height_meters = estimate_building_height(shadow_length_meters, config.solar_elevation_angle)
            start_point = shadow_info["shadow_start_point"]
            end_point = shadow_info["shadow_end_point"]

        result_item = {
            "asset_index": building["asset_index"],
            "class": "building",
            "bbox": bbox,
            "shadow_contour": matched_shadow,
            "shadow_start_point": start_point,
            "shadow_end_point": end_point,
            "shadow_length_pixels": round(max(shadow_length_pixels, 0.0), 2),
            "shadow_length_meters": round(max(shadow_length_meters, 0.0), 2),
            "estimated_height_meters": round(max(height_meters, 0.0), 2),
        }
        building_results.append(result_item)

        # Extend existing JSON-like asset structure in place.
        asset = detected_assets[building["asset_index"]]
        asset["shadow_length_pixels"] = result_item["shadow_length_pixels"]
        asset["shadow_length_meters"] = result_item["shadow_length_meters"]
        asset["estimated_height_meters"] = result_item["estimated_height_meters"]

    return detected_assets, building_results, shadow_mask


def draw_and_save_outputs(
    analyzed_base_bgr: Any,
    building_results: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, str]:
    """Save `analyzed_output.jpg` and `results.json` exactly as requested."""
    import cv2  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_image = draw_height_overlay(analyzed_base_bgr, building_results)
    analyzed_output_path = output_dir / "analyzed_output.jpg"
    json_output_path = output_dir / "results.json"

    cv2.imwrite(str(analyzed_output_path), overlay_image)

    export_rows: List[Dict[str, Any]] = []
    for item in building_results:
        x1, y1, x2, y2 = item["bbox"]
        export_rows.append(
            {
                "class": "building",
                "bbox": [x1, y1, x2, y2],
                "shadow_length_pixels": item["shadow_length_pixels"],
                "shadow_length_meters": item["shadow_length_meters"],
                "estimated_height_meters": item["estimated_height_meters"],
            }
        )

    json_output_path.write_text(json.dumps(export_rows, indent=2), encoding="utf-8")
    return {
        "analyzed_output_path": str(analyzed_output_path),
        "results_json_path": str(json_output_path),
        "overlay_data_url": _encode_bgr_to_data_url(overlay_image, ".jpg"),
    }


def apply_height_estimation_to_analysis(
    image_path: str,
    validated_response: Dict[str, Any],
    response_extras: Dict[str, Any] | None = None,
    request_id: str | None = None,
    output_root: str | None = None,
    meters_per_pixel: float = 0.2,
    solar_elevation_angle: float = 45.0,
) -> Dict[str, Any]:
    """Attach building height estimation to existing pipeline output.

    The function:
    1. Reuses existing building detections from `validated_response`.
    2. Detects shadows on the source image.
    3. Computes height estimates and injects them into JSON.
    4. Draws overlays on the already analyzed image (when available).
    5. Saves `analyzed_output.jpg` and `results.json`.
    """
    import cv2  # type: ignore

    result_extras = dict(response_extras or {})
    warnings: List[str] = []

    source_bgr = cv2.imread(image_path)
    if source_bgr is None:
        warnings.append("Height estimation skipped: unable to load source image.")
        return {
            "validated_response": validated_response,
            "response_extras": result_extras,
            "warnings": warnings,
        }

    analyzed_base = None
    for key in ("height_visualization_data_url", "naming_visualization_data_url", "asset_map_visualization_data_url"):
        candidate = str(result_extras.get(key, "")).strip()
        if not candidate:
            continue
        decoded = _decode_data_url_to_bgr(candidate)
        if decoded is not None:
            analyzed_base = decoded
            break
    if analyzed_base is None:
        analyzed_base = source_bgr.copy()

    config = HeightEstimationConfig(
        meters_per_pixel=max(float(meters_per_pixel), 1e-6),
        solar_elevation_angle=max(min(float(solar_elevation_angle), 89.9), 1e-3),
    )

    detected_assets = list(validated_response.get("detected_assets", []))
    updated_assets, building_results, _ = detect_shadows_for_buildings(source_bgr, detected_assets, config)
    validated_response["detected_assets"] = updated_assets

    output_base = Path(output_root or (Path.cwd() / "height_estimation_outputs"))
    run_id = request_id or uuid.uuid4().hex
    run_dir = output_base / run_id
    save_info = draw_and_save_outputs(analyzed_base, building_results, run_dir)

    if config.debug_matplotlib:
        overlay_bgr = _decode_data_url_to_bgr(save_info["overlay_data_url"])
        if overlay_bgr is not None:
            matplotlib_preview(overlay_bgr, title="Shadow-based Building Height Estimation")

    result_extras["height_visualization_data_url"] = save_info["overlay_data_url"]
    result_extras["height_estimation_output_path"] = save_info["analyzed_output_path"]
    result_extras["height_estimation_json_path"] = save_info["results_json_path"]
    result_extras["height_estimation_config"] = {
        "meters_per_pixel": config.meters_per_pixel,
        "solar_elevation_angle": config.solar_elevation_angle,
    }

    warnings.append(
        (
            f"Height estimation applied using meters_per_pixel={config.meters_per_pixel} "
            f"and solar_elevation_angle={config.solar_elevation_angle}."
        )
    )
    return {
        "validated_response": validated_response,
        "response_extras": result_extras,
        "warnings": warnings,
    }


def run_single_image_height_estimation(
    image_path: str,
    detected_assets: List[Dict[str, Any]],
    analyzed_image_path: str | None = None,
    output_dir: str = "./height_estimation_outputs",
    meters_per_pixel: float = 0.2,
    solar_elevation_angle: float = 45.0,
) -> Dict[str, Any]:
    """Notebook-friendly helper for Kaggle/local experiments."""
    import cv2  # type: ignore

    source_bgr = cv2.imread(image_path)
    if source_bgr is None:
        raise FileNotFoundError(f"Unable to open image: {image_path}")

    analyzed_base = source_bgr.copy()
    if analyzed_image_path:
        analyzed_loaded = cv2.imread(analyzed_image_path)
        if analyzed_loaded is not None:
            analyzed_base = analyzed_loaded

    config = HeightEstimationConfig(
        meters_per_pixel=max(float(meters_per_pixel), 1e-6),
        solar_elevation_angle=max(min(float(solar_elevation_angle), 89.9), 1e-3),
    )

    updated_assets, building_results, _ = detect_shadows_for_buildings(source_bgr, detected_assets, config)
    save_info = draw_and_save_outputs(analyzed_base, building_results, Path(output_dir))
    return {
        "detected_assets": updated_assets,
        "building_height_results": building_results,
        "outputs": save_info,
    }

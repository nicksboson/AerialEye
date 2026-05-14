from __future__ import annotations

from typing import Any, Dict, List, Tuple

BUILDING_CATEGORIES = {
    "properties & buildings",
    "properties and buildings",
    "building",
    "buildings",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_building_category(raw_category: Any) -> bool:
    category = str(raw_category or "").strip().lower()
    return category in BUILDING_CATEGORIES or "building" in category


def _bbox_from_asset(asset: Dict[str, Any], image_w: int, image_h: int) -> Tuple[int, int, int, int] | None:
    bbox = asset.get("bounding_box")
    if isinstance(bbox, dict):
        x_min = _safe_float(bbox.get("x_min"), 0.0)
        y_min = _safe_float(bbox.get("y_min"), 0.0)
        x_max = _safe_float(bbox.get("x_max"), 0.0)
        y_max = _safe_float(bbox.get("y_max"), 0.0)

        # The main pipeline uses 0..100 normalized coordinates.
        if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) <= 100.0:
            x_min = (x_min / 100.0) * image_w
            x_max = (x_max / 100.0) * image_w
            y_min = (y_min / 100.0) * image_h
            y_max = (y_max / 100.0) * image_h
    elif isinstance(bbox, list) and len(bbox) == 4:
        # Compatibility with generic [x1, y1, x2, y2] format.
        x_min = _safe_float(bbox[0], 0.0)
        y_min = _safe_float(bbox[1], 0.0)
        x_max = _safe_float(bbox[2], 0.0)
        y_max = _safe_float(bbox[3], 0.0)
    else:
        return None

    x1 = int(round(max(0.0, min(float(image_w - 1), min(x_min, x_max)))))
    y1 = int(round(max(0.0, min(float(image_h - 1), min(y_min, y_max)))))
    x2 = int(round(max(0.0, min(float(image_w - 1), max(x_min, x_max)))))
    y2 = int(round(max(0.0, min(float(image_h - 1), max(y_min, y_max)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def detect_buildings(detected_assets: List[Dict[str, Any]], image_shape: Tuple[int, int, int] | Tuple[int, int]) -> List[Dict[str, Any]]:
    """Extract building detections from already-generated pipeline asset metadata.

    This function intentionally reuses existing detections and does not invoke an extra detector.
    """
    if len(image_shape) < 2:
        return []
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])

    buildings: List[Dict[str, Any]] = []
    for index, asset in enumerate(detected_assets):
        if not isinstance(asset, dict):
            continue
        if not _is_building_category(asset.get("category")):
            continue
        bbox = _bbox_from_asset(asset, image_w, image_h)
        if bbox is None:
            continue

        buildings.append(
            {
                "asset_index": index,
                "unique_id": str(asset.get("unique_id", f"building_{index+1}")),
                "class": "building",
                "bbox": bbox,
            }
        )

    return buildings

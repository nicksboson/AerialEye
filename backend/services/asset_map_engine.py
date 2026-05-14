import base64
import logging
from typing import Any, Dict, List, Tuple

from backend.services.area_utils import get_meters_per_pixel, pixel_area_to_sq_m, pixel_length_to_m
from backend.services.vision_engine import (
    AnalysisResult,
    VisionEngineError,
    _build_transformed_payload,
    default_response_schema,
    validate_response_schema,
)

logger = logging.getLogger(__name__)

ASSET_TO_CATEGORY = {
    "properties_buildings": "Properties & Buildings",
    "trees_green_cover": "Trees & Green Cover",
    "parks_open_spaces": "Parks & Open Spaces",
    "water_bodies": "Water Bodies",
    "roads_footpaths": "Roads & Footpaths",
    "drains_sewage": "Drains & Sewage",
    "vehicles_parking": "Vehicles & Parking",
    "waste_dumps": "Waste Dumps",
    "solar_panels": "Solar Panels",
}

ASSET_COLORS_RGB = {
    "properties_buildings": (255, 0, 0),
    "trees_green_cover": (0, 255, 0),
    "parks_open_spaces": (120, 255, 120),
    "water_bodies": (0, 80, 255),
    "roads_footpaths": (180, 120, 70),
    "drains_sewage": (0, 0, 0),
    "vehicles_parking": (255, 255, 0),
    "waste_dumps": (255, 120, 0),
    "solar_panels": (80, 0, 160),
}

MIN_CONTOUR_AREA = {
    "properties_buildings": 80.0,
    "trees_green_cover": 120.0,
    "parks_open_spaces": 800.0,
    "water_bodies": 120.0,
    "roads_footpaths": 180.0,
    "drains_sewage": 80.0,
    "vehicles_parking": 18.0,
    "waste_dumps": 80.0,
    "solar_panels": 20.0,
}


def _clean_mask(mask: Any, np_mod: Any, cv2_mod: Any, k: int = 5) -> Any:
    kernel = np_mod.ones((k, k), np_mod.uint8)
    cleaned = cv2_mod.morphologyEx(mask, cv2_mod.MORPH_OPEN, kernel)
    cleaned = cv2_mod.morphologyEx(cleaned, cv2_mod.MORPH_CLOSE, kernel)
    return cleaned


def _norm(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    scaled = (value / maximum) * 100.0
    if scaled < 0.0:
        return 0.0
    if scaled > 100.0:
        return 100.0
    return scaled


def _bbox_polygon(x_min: float, y_min: float, x_max: float, y_max: float) -> List[Dict[str, float]]:
    return [
        {"x": round(x_min, 4), "y": round(y_min, 4)},
        {"x": round(x_max, 4), "y": round(y_min, 4)},
        {"x": round(x_max, 4), "y": round(y_max, 4)},
        {"x": round(x_min, 4), "y": round(y_max, 4)},
    ]


def _mask_to_assets(
    mask: Any,
    asset_key: str,
    image_w: int,
    image_h: int,
    cv2_mod: Any,
    detection_index_start: int,
    meters_per_pixel: float,
) -> Tuple[List[Dict[str, Any]], int]:
    contours, _ = cv2_mod.findContours(mask, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
    next_index = detection_index_start
    items: List[Dict[str, Any]] = []
    image_area = max(float(image_w * image_h), 1.0)

    for cnt in contours:
        area_px = float(cv2_mod.contourArea(cnt))
        if area_px < MIN_CONTOUR_AREA.get(asset_key, 50.0):
            continue

        x, y, bw, bh = cv2_mod.boundingRect(cnt)
        x_min = _norm(float(x), float(image_w))
        y_min = _norm(float(y), float(image_h))
        x_max = _norm(float(x + bw), float(image_w))
        y_max = _norm(float(y + bh), float(image_h))
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        approx = cv2_mod.approxPolyDP(cnt, 0.01 * cv2_mod.arcLength(cnt, True), True)
        polygon: List[Dict[str, float]] = []
        for point in approx:
            px, py = point[0]
            polygon.append(
                {
                    "x": round(_norm(float(px), float(image_w)), 4),
                    "y": round(_norm(float(py), float(image_h)), 4),
                }
            )
        if len(polygon) < 3:
            polygon = _bbox_polygon(x_min, y_min, x_max, y_max)

        coverage_percent = (area_px / image_area) * 100.0
        confidence = min(99.0, max(45.0, 60.0 + coverage_percent * 4.0))
        category = ASSET_TO_CATEGORY[asset_key]

        items.append(
            {
                "unique_id": f"asset_map_{next_index}",
                "category": category,
                "subcategory": asset_key,
                "confidence_percent": round(confidence, 2),
                "estimated_count": 1,
                "estimated_area_sq_m": round(pixel_area_to_sq_m(area_px, meters_per_pixel), 2),
                "estimated_dimensions_m": {
                    "length": round(pixel_length_to_m(float(bw), meters_per_pixel), 2),
                    "width": round(pixel_length_to_m(float(bh), meters_per_pixel), 2),
                },
                "estimated_coverage_percent": round(min(100.0, max(0.0, coverage_percent)), 4),
                "condition_status": "Monitored",
                "maintenance_priority": "Medium",
                "center_coordinates": {"x": round(center_x, 4), "y": round(center_y, 4)},
                "bounding_box": {
                    "x_min": round(x_min, 4),
                    "y_min": round(y_min, 4),
                    "x_max": round(x_max, 4),
                    "y_max": round(y_max, 4),
                },
                "polygon_coordinates": polygon,
                "visual_description": f"{asset_key} detected by CV asset mapping pipeline.",
                "estimation_basis": "OpenCV thresholding + contour extraction from uploaded image.",
            }
        )
        next_index += 1

    return items, next_index


def _build_masks(image_path: str, np_mod: Any, cv2_mod: Any) -> Tuple[Dict[str, Any], Any, int, int]:
    img_bgr = cv2_mod.imread(image_path)
    if img_bgr is None:
        raise VisionEngineError(f"Unable to read input image: {image_path}")

    img_rgb = cv2_mod.cvtColor(img_bgr, cv2_mod.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    hsv = cv2_mod.cvtColor(img_rgb, cv2_mod.COLOR_RGB2HSV)
    gray = cv2_mod.cvtColor(img_rgb, cv2_mod.COLOR_RGB2GRAY)

    kernel3 = np_mod.ones((3, 3), np_mod.uint8)
    kernel9 = np_mod.ones((9, 9), np_mod.uint8)

    green_mask = cv2_mod.inRange(hsv, np_mod.array([30, 35, 25]), np_mod.array([95, 255, 255]))
    green_mask = _clean_mask(green_mask, np_mod, cv2_mod, 5)

    water_mask = cv2_mod.inRange(hsv, np_mod.array([85, 20, 10]), np_mod.array([135, 255, 190]))
    water_mask = cv2_mod.bitwise_and(water_mask, cv2_mod.bitwise_not(green_mask))
    water_mask = _clean_mask(water_mask, np_mod, cv2_mod, 5)

    road_color_mask = cv2_mod.inRange(hsv, np_mod.array([5, 10, 75]), np_mod.array([35, 140, 245]))
    edges = cv2_mod.Canny(gray, 60, 160)
    edges = cv2_mod.dilate(edges, kernel3, iterations=1)
    road_mask = cv2_mod.bitwise_or(road_color_mask, edges)
    road_mask = cv2_mod.bitwise_and(road_mask, cv2_mod.bitwise_not(green_mask))
    road_mask = cv2_mod.bitwise_and(road_mask, cv2_mod.bitwise_not(water_mask))
    road_mask = _clean_mask(road_mask, np_mod, cv2_mod, 3)

    bright_mask = cv2_mod.inRange(gray, 120, 255)
    low_green = cv2_mod.bitwise_not(green_mask)
    building_mask = cv2_mod.bitwise_and(bright_mask, low_green)
    building_mask = cv2_mod.bitwise_and(building_mask, cv2_mod.bitwise_not(water_mask))
    building_mask = cv2_mod.bitwise_and(building_mask, cv2_mod.bitwise_not(road_mask))
    building_mask = _clean_mask(building_mask, np_mod, cv2_mod, 3)

    building_filtered = np_mod.zeros_like(building_mask)
    building_contours, _ = cv2_mod.findContours(building_mask, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
    for cnt in building_contours:
        area = cv2_mod.contourArea(cnt)
        if area < 80 or area > 20000:
            continue
        x, y, bw, bh = cv2_mod.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        rect_area = bw * bh
        extent = area / max(rect_area, 1)
        if 0.25 < aspect < 4.0 and extent > 0.25:
            cv2_mod.drawContours(building_filtered, [cnt], -1, 255, -1)
    building_mask = building_filtered

    parks_mask = cv2_mod.morphologyEx(green_mask, cv2_mod.MORPH_CLOSE, kernel9)
    parks_filtered = np_mod.zeros_like(parks_mask)
    parks_contours, _ = cv2_mod.findContours(parks_mask, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
    for cnt in parks_contours:
        if cv2_mod.contourArea(cnt) > 5000:
            cv2_mod.drawContours(parks_filtered, [cnt], -1, 255, -1)
    parks_mask = parks_filtered

    dark_linear = cv2_mod.inRange(gray, 0, 90)
    near_road = cv2_mod.dilate(road_mask, kernel9, iterations=1)
    near_water = cv2_mod.dilate(water_mask, kernel9, iterations=1)
    drain_mask = cv2_mod.bitwise_and(dark_linear, cv2_mod.bitwise_or(near_road, near_water))
    drain_mask = cv2_mod.bitwise_and(drain_mask, cv2_mod.bitwise_not(green_mask))
    drain_mask = _clean_mask(drain_mask, np_mod, cv2_mod, 3)

    vehicle_mask = np_mod.zeros_like(gray)
    near_road_small = cv2_mod.dilate(road_mask, kernel9, iterations=1)
    candidate_vehicle = cv2_mod.bitwise_and(bright_mask, near_road_small)
    vehicle_contours, _ = cv2_mod.findContours(
        candidate_vehicle, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE
    )
    for cnt in vehicle_contours:
        area = cv2_mod.contourArea(cnt)
        if area < 8 or area > 120:
            continue
        _, _, bw, bh = cv2_mod.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if 0.4 < aspect < 3.5:
            cv2_mod.drawContours(vehicle_mask, [cnt], -1, 255, -1)

    waste_color = cv2_mod.inRange(hsv, np_mod.array([8, 20, 80]), np_mod.array([35, 180, 230]))
    waste_mask = cv2_mod.bitwise_and(waste_color, cv2_mod.bitwise_not(road_mask))
    waste_mask = cv2_mod.bitwise_and(waste_mask, cv2_mod.bitwise_not(building_mask))
    waste_mask = _clean_mask(waste_mask, np_mod, cv2_mod, 5)

    solar_color = cv2_mod.inRange(hsv, np_mod.array([95, 20, 20]), np_mod.array([135, 180, 120]))
    solar_candidates = cv2_mod.bitwise_and(solar_color, cv2_mod.dilate(building_mask, kernel9, iterations=1))
    solar_mask = _clean_mask(solar_candidates, np_mod, cv2_mod, 3)

    masks = {
        "properties_buildings": building_mask,
        "trees_green_cover": green_mask,
        "parks_open_spaces": parks_mask,
        "water_bodies": water_mask,
        "roads_footpaths": road_mask,
        "drains_sewage": drain_mask,
        "vehicles_parking": vehicle_mask,
        "waste_dumps": waste_mask,
        "solar_panels": solar_mask,
    }

    overlay = img_rgb.copy()
    for key, mask in masks.items():
        color = np_mod.array(ASSET_COLORS_RGB[key], dtype=np_mod.uint8)
        overlay[mask > 0] = color
    blended = cv2_mod.addWeighted(img_rgb, 0.45, overlay, 0.55, 0)

    return masks, blended, w, h


def _overlay_to_data_url(blended_rgb: Any, cv2_mod: Any) -> str:
    blended_bgr = cv2_mod.cvtColor(blended_rgb, cv2_mod.COLOR_RGB2BGR)
    success, encoded = cv2_mod.imencode(".png", blended_bgr)
    if not success:
        raise VisionEngineError("Failed to encode asset map overlay image.")
    data = encoded.tobytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"


def analyze_asset_map_image(image_path: str) -> AnalysisResult:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as err:
        raise VisionEngineError(
            "OpenCV/Numpy dependencies are missing. Run `pip install -r requirements.txt`."
        ) from err

    masks, blended_rgb, image_w, image_h = _build_masks(image_path, np, cv2)
    image_area = max(float(image_w * image_h), 1.0)
    meters_per_pixel = get_meters_per_pixel()

    detected_assets: List[Dict[str, Any]] = []
    category_statistics: Dict[str, Dict[str, float]] = {
        category: {"count": 0, "total_area_sq_m": 0.0, "coverage_percent": 0.0}
        for category in ASSET_TO_CATEGORY.values()
    }

    detection_index = 1
    for asset_key, mask in masks.items():
        category = ASSET_TO_CATEGORY[asset_key]
        items, detection_index = _mask_to_assets(
            mask,
            asset_key,
            image_w,
            image_h,
            cv2,
            detection_index,
            meters_per_pixel,
        )
        detected_assets.extend(items)

        pixel_count = int(np.sum(mask > 0))
        coverage = (pixel_count / image_area) * 100.0
        category_statistics[category] = {
            "count": len(items),
            "total_area_sq_m": round(pixel_area_to_sq_m(float(pixel_count), meters_per_pixel), 2),
            "coverage_percent": round(min(100.0, max(0.0, coverage)), 4),
        }

    total_assets = len(detected_assets)
    avg_conf = 0.0
    if detected_assets:
        avg_conf = sum(float(item.get("confidence_percent", 0.0)) for item in detected_assets) / total_assets

    green_percent = (
        category_statistics["Trees & Green Cover"]["coverage_percent"]
        + category_statistics["Parks & Open Spaces"]["coverage_percent"]
    )
    road_percent = category_statistics["Roads & Footpaths"]["coverage_percent"]
    water_percent = category_statistics["Water Bodies"]["coverage_percent"]
    built_up_percent = category_statistics["Properties & Buildings"]["coverage_percent"]

    green_status = "High" if green_percent >= 60 else ("Moderate" if green_percent >= 30 else "Low")
    road_status = "High" if road_percent >= 10 else ("Moderate" if road_percent >= 4 else "Low")
    risk_score = "Moderate" if water_percent >= 5 else "Low"

    payload = default_response_schema()
    payload["image_analysis"] = {
        "scene_type": "CV Asset Map Analysis",
        "image_quality": "Processed",
        "estimated_total_area_sq_m": round(pixel_area_to_sq_m(image_area, meters_per_pixel), 2),
        "overall_detection_confidence_percent": round(min(100.0, max(0.0, avg_conf)), 2),
        "dominant_land_use": "Mixed Urban",
    }
    payload["summary_statistics"] = {
        "total_assets_detected": total_assets,
        "green_cover_percent": round(min(100.0, max(0.0, green_percent)), 2),
        "built_up_percent": round(min(100.0, max(0.0, built_up_percent)), 2),
        "road_network_percent": round(min(100.0, max(0.0, road_percent)), 2),
        "water_body_percent": round(min(100.0, max(0.0, water_percent)), 2),
        "open_space_percent": round(
            min(100.0, max(0.0, category_statistics["Parks & Open Spaces"]["coverage_percent"])),
            2,
        ),
    }
    payload["detected_assets"] = detected_assets
    payload["category_statistics"] = category_statistics
    payload["risk_analysis"] = {
        "encroachment_risk": "Moderate" if built_up_percent > 40 else "Low",
        "drainage_risk": "Moderate" if category_statistics["Drains & Sewage"]["count"] > 0 else "Low",
        "environmental_risk": "Moderate" if green_percent < 25 else "Low",
        "flood_risk": risk_score,
    }
    payload["ai_insights"] = [
        f"Green density is {green_status} ({green_percent:.2f}%).",
        f"Road density is {road_status} ({road_percent:.2f}%).",
        f"Urban risk score inferred from CV masks is {risk_score}.",
    ]

    validated, validation_warnings = validate_response_schema(payload)
    transformed = _build_transformed_payload(validated)

    overlay_url = _overlay_to_data_url(blended_rgb, cv2)
    response_extras = {
        "asset_map_visualization_data_url": overlay_url,
    }

    warnings: List[str] = []
    warnings.extend(validation_warnings)
    warnings.append(f"Area estimates use METERS_PER_PIXEL={meters_per_pixel}.")

    return AnalysisResult(
        validated_response=validated,
        transformed=transformed,
        raw_text=f"cv_asset_map_detected_assets={total_assets}",
        warnings=warnings,
        response_extras=response_extras,
    )

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.services.area_utils import get_meters_per_pixel, pixel_area_to_sq_m
from backend.services.vision_engine import (
    AnalysisResult,
    VisionEngineError,
    _build_transformed_payload,
    default_response_schema,
    validate_response_schema,
)

CATEGORY_ORDER = [
    "Properties & Buildings",
    "Trees & Green Cover",
    "Parks & Open Spaces",
    "Water Bodies",
    "Roads & Footpaths",
    "Drains & Sewage",
    "Vehicles & Parking",
    "Waste Dumps",
    "Solar Panels",
]

COLORS = {
    "Trees & Green Cover": (0, 255, 0),
    "Parks & Open Spaces": (80, 220, 80),
    "Water Bodies": (255, 80, 0),
    "Roads & Footpaths": (70, 130, 200),
    "Drains & Sewage": (30, 30, 30),
    "Properties & Buildings": (0, 0, 255),
    "Vehicles & Parking": (0, 255, 255),
    "Waste Dumps": (0, 140, 255),
    "Solar Panels": (180, 0, 180),
}


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return max(default, min_value)
    return max(value, min_value)


def _clean_mask(mask: Any, np_mod: Any, cv2_mod: Any, k: int = 5) -> Any:
    kernel = np_mod.ones((k, k), np_mod.uint8)
    mask = cv2_mod.morphologyEx(mask, cv2_mod.MORPH_OPEN, kernel)
    mask = cv2_mod.morphologyEx(mask, cv2_mod.MORPH_CLOSE, kernel)
    return mask


def _filter_contours(
    mask: Any,
    cv2_mod: Any,
    np_mod: Any,
    min_area: float = 100,
    max_area: float | None = None,
    min_aspect: float | None = None,
    max_aspect: float | None = None,
    min_extent: float | None = None,
) -> Tuple[Any, List[Tuple[Any, float, int, int, int, int]]]:
    out = np_mod.zeros_like(mask)
    kept: List[Tuple[Any, float, int, int, int, int]] = []
    contours, _ = cv2_mod.findContours(mask, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = float(cv2_mod.contourArea(cnt))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        x, y, w, h = cv2_mod.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        aspect = max(float(w), float(h)) / max(float(min(w, h)), 1.0)
        extent = area / max(float(w * h), 1.0)

        if min_aspect is not None and aspect < min_aspect:
            continue
        if max_aspect is not None and aspect > max_aspect:
            continue
        if min_extent is not None and extent < min_extent:
            continue

        cv2_mod.drawContours(out, [cnt], -1, 255, -1)
        kept.append((cnt, area, x, y, w, h))

    return out, kept


def _draw_mask_objects(
    frame: Any,
    mask: Any,
    label: str,
    color: Tuple[int, int, int],
    cv2_mod: Any,
    min_area: float = 250,
) -> Tuple[int, float]:
    contours, _ = cv2_mod.findContours(mask, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
    count = 0
    area_sum = 0.0

    for cnt in contours:
        area = float(cv2_mod.contourArea(cnt))
        if area < min_area:
            continue

        x, y, w, h = cv2_mod.boundingRect(cnt)
        count += 1
        area_sum += area

        cv2_mod.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2_mod.putText(
            frame,
            label,
            (x, max(y - 6, 16)),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2_mod.LINE_AA,
        )

    return count, area_sum


def _draw_info_panel(
    frame: Any,
    counts: Dict[str, int],
    area_pixels: Dict[str, float],
    total_pixels: int,
    cv2_mod: Any,
) -> None:
    panel_lines = ["GEO VISION AI ANALYTICS"]
    safe_total = max(float(total_pixels), 1.0)

    for label in CATEGORY_ORDER:
        count = int(counts.get(label, 0))
        area_px = float(area_pixels.get(label, 0.0))
        if count <= 0 and area_px <= 0:
            continue
        coverage_pct = (area_px / safe_total) * 100.0
        panel_lines.append(f"{label}: {count} | {coverage_pct:.1f}%")

    h = frame.shape[0]
    panel_h = min(28 + 22 * len(panel_lines), h)
    cv2_mod.rectangle(frame, (0, 0), (520, panel_h), (20, 20, 20), -1)

    y = 24
    for line in panel_lines:
        cv2_mod.putText(
            frame,
            line,
            (12, y),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2_mod.LINE_AA,
        )
        y += 22


def _detect_assets_in_frame(frame_bgr: Any, cv2_mod: Any, np_mod: Any) -> Tuple[Any, Dict[str, int], Dict[str, float], int]:
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    total_pixels = int(h * w)

    hsv = cv2_mod.cvtColor(frame_bgr, cv2_mod.COLOR_BGR2HSV)
    gray = cv2_mod.cvtColor(frame_bgr, cv2_mod.COLOR_BGR2GRAY)

    kernel7 = np_mod.ones((7, 7), np_mod.uint8)
    kernel11 = np_mod.ones((11, 11), np_mod.uint8)

    masks: Dict[str, Any] = {}

    green_mask = cv2_mod.inRange(hsv, np_mod.array([30, 40, 25]), np_mod.array([95, 255, 255]))
    green_mask = _clean_mask(green_mask, np_mod, cv2_mod, 5)
    green_area_pct = float(np_mod.sum(green_mask > 0)) / max(float(total_pixels), 1.0) * 100.0
    if green_area_pct > 1.0:
        masks["Trees & Green Cover"] = green_mask

    parks_mask = cv2_mod.morphologyEx(green_mask, cv2_mod.MORPH_CLOSE, kernel11)
    parks_mask, parks_contours = _filter_contours(parks_mask, cv2_mod, np_mod, min_area=8000)
    if parks_contours:
        masks["Parks & Open Spaces"] = parks_mask

    water_mask = cv2_mod.inRange(hsv, np_mod.array([85, 25, 10]), np_mod.array([135, 255, 180]))
    water_mask = cv2_mod.bitwise_and(water_mask, cv2_mod.bitwise_not(green_mask))
    water_mask = _clean_mask(water_mask, np_mod, cv2_mod, 5)
    water_mask, water_contours = _filter_contours(water_mask, cv2_mod, np_mod, min_area=1500)
    if water_contours:
        masks["Water Bodies"] = water_mask

    road_color = cv2_mod.inRange(hsv, np_mod.array([5, 10, 55]), np_mod.array([35, 150, 245]))
    low_sat_gray = cv2_mod.inRange(hsv, np_mod.array([0, 0, 65]), np_mod.array([180, 55, 230]))
    road_candidate = cv2_mod.bitwise_or(road_color, low_sat_gray)
    road_candidate = cv2_mod.bitwise_and(road_candidate, cv2_mod.bitwise_not(green_mask))
    if "Water Bodies" in masks:
        road_candidate = cv2_mod.bitwise_and(road_candidate, cv2_mod.bitwise_not(masks["Water Bodies"]))
    road_candidate = _clean_mask(road_candidate, np_mod, cv2_mod, 5)
    road_mask, road_contours = _filter_contours(
        road_candidate,
        cv2_mod,
        np_mod,
        min_area=2500,
        min_aspect=3.0,
        min_extent=0.18,
    )
    if road_contours:
        masks["Roads & Footpaths"] = road_mask

    bright = cv2_mod.inRange(gray, 135, 255)
    roof_low_sat = cv2_mod.inRange(hsv, np_mod.array([0, 0, 90]), np_mod.array([180, 80, 255]))
    building_candidate = cv2_mod.bitwise_and(bright, roof_low_sat)
    building_candidate = cv2_mod.bitwise_and(building_candidate, cv2_mod.bitwise_not(green_mask))
    if "Roads & Footpaths" in masks:
        building_candidate = cv2_mod.bitwise_and(building_candidate, cv2_mod.bitwise_not(masks["Roads & Footpaths"]))
    if "Water Bodies" in masks:
        building_candidate = cv2_mod.bitwise_and(building_candidate, cv2_mod.bitwise_not(masks["Water Bodies"]))
    building_candidate = _clean_mask(building_candidate, np_mod, cv2_mod, 3)
    building_mask, building_contours = _filter_contours(
        building_candidate,
        cv2_mod,
        np_mod,
        min_area=220,
        max_area=18000,
        max_aspect=4.5,
        min_extent=0.45,
    )
    building_area_pct = float(np_mod.sum(building_mask > 0)) / max(float(total_pixels), 1.0) * 100.0
    if len(building_contours) >= 2 and building_area_pct > 0.05:
        masks["Properties & Buildings"] = building_mask

    dark = cv2_mod.inRange(gray, 0, 75)
    drain_context = np_mod.zeros_like(gray)
    if "Roads & Footpaths" in masks:
        drain_context = cv2_mod.bitwise_or(
            drain_context,
            cv2_mod.dilate(masks["Roads & Footpaths"], kernel11, iterations=1),
        )
    if "Water Bodies" in masks:
        drain_context = cv2_mod.bitwise_or(
            drain_context,
            cv2_mod.dilate(masks["Water Bodies"], kernel11, iterations=1),
        )
    drain_candidate = cv2_mod.bitwise_and(dark, drain_context)
    drain_candidate = cv2_mod.bitwise_and(drain_candidate, cv2_mod.bitwise_not(green_mask))
    drain_candidate = _clean_mask(drain_candidate, np_mod, cv2_mod, 3)
    drain_mask, drain_contours = _filter_contours(
        drain_candidate,
        cv2_mod,
        np_mod,
        min_area=180,
        max_area=8000,
        min_aspect=4.0,
        min_extent=0.12,
    )
    if drain_contours:
        masks["Drains & Sewage"] = drain_mask

    if "Roads & Footpaths" in masks:
        vehicle_candidate = cv2_mod.bitwise_and(bright, cv2_mod.dilate(masks["Roads & Footpaths"], kernel7, iterations=1))
        vehicle_candidate = cv2_mod.bitwise_and(vehicle_candidate, cv2_mod.bitwise_not(green_mask))
        vehicle_candidate = _clean_mask(vehicle_candidate, np_mod, cv2_mod, 2)
        vehicle_mask, vehicle_contours = _filter_contours(
            vehicle_candidate,
            cv2_mod,
            np_mod,
            min_area=12,
            max_area=180,
            max_aspect=4.0,
            min_extent=0.35,
        )
        if len(vehicle_contours) >= 3:
            masks["Vehicles & Parking"] = vehicle_mask

    waste_candidate = cv2_mod.inRange(hsv, np_mod.array([8, 35, 80]), np_mod.array([35, 210, 245]))
    if "Roads & Footpaths" in masks:
        waste_candidate = cv2_mod.bitwise_and(waste_candidate, cv2_mod.bitwise_not(masks["Roads & Footpaths"]))
    if "Properties & Buildings" in masks:
        waste_candidate = cv2_mod.bitwise_and(waste_candidate, cv2_mod.bitwise_not(masks["Properties & Buildings"]))
    waste_candidate = cv2_mod.bitwise_and(waste_candidate, cv2_mod.bitwise_not(green_mask))
    waste_candidate = _clean_mask(waste_candidate, np_mod, cv2_mod, 5)
    waste_mask, waste_contours = _filter_contours(
        waste_candidate,
        cv2_mod,
        np_mod,
        min_area=2000,
        max_area=50000,
        min_extent=0.15,
    )
    if waste_contours:
        masks["Waste Dumps"] = waste_mask

    if "Properties & Buildings" in masks:
        solar_candidate = cv2_mod.inRange(hsv, np_mod.array([95, 25, 15]), np_mod.array([135, 180, 130]))
        solar_candidate = cv2_mod.bitwise_and(
            solar_candidate,
            cv2_mod.dilate(masks["Properties & Buildings"], kernel7, iterations=1),
        )
        solar_candidate = _clean_mask(solar_candidate, np_mod, cv2_mod, 3)
        solar_mask, solar_contours = _filter_contours(
            solar_candidate,
            cv2_mod,
            np_mod,
            min_area=80,
            max_area=4000,
            max_aspect=5.0,
            min_extent=0.40,
        )
        if solar_contours:
            masks["Solar Panels"] = solar_mask

    overlay = frame.copy()
    for label, mask in masks.items():
        overlay[mask > 0] = COLORS[label]
    frame = cv2_mod.addWeighted(frame, 0.70, overlay, 0.30, 0)

    counts: Dict[str, int] = {}
    area_pixels: Dict[str, float] = {}

    draw_config = {
        "Trees & Green Cover": ("Green Cover", 1200),
        "Parks & Open Spaces": ("Park/Open Space", 8000),
        "Water Bodies": ("Water Body", 1500),
        "Roads & Footpaths": ("Road/Footpath", 2500),
        "Drains & Sewage": ("Drain/Sewage", 180),
        "Properties & Buildings": ("Building", 220),
        "Vehicles & Parking": ("Vehicle/Parking", 12),
        "Waste Dumps": ("Waste Dump", 2000),
        "Solar Panels": ("Solar Panel", 80),
    }

    for label, mask in masks.items():
        short_label, min_area = draw_config[label]
        count, _ = _draw_mask_objects(frame, mask, short_label, COLORS[label], cv2_mod, min_area=min_area)
        counts[label] = count
        pixel_count = float(np_mod.sum(mask > 0))
        area_pixels[label] = pixel_count

    _draw_info_panel(frame, counts, area_pixels, total_pixels, cv2_mod)

    return frame, counts, area_pixels, total_pixels


def _dominant_land_use(category_coverage: Dict[str, float]) -> str:
    active = [(name, coverage) for name, coverage in category_coverage.items() if coverage > 0]
    if not active:
        return "Unknown"
    active.sort(key=lambda item: item[1], reverse=True)
    return active[0][0]


def _prepare_detection_frame(frame: Any, cv2_mod: Any, max_dim: int) -> Tuple[Any, float]:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if max_dim <= 0 or longest <= max_dim:
        return frame, 1.0

    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2_mod.resize(frame, (new_w, new_h), interpolation=cv2_mod.INTER_AREA)
    return resized, scale


def analyze_asset_video(video_path: str, output_dir: str) -> AnalysisResult:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as err:
        raise VisionEngineError(
            "OpenCV/Numpy dependencies are missing. Run `pip install -r requirements.txt`."
        ) from err

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VisionEngineError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 12.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise VisionEngineError("Invalid video dimensions from uploaded file.")

    # Performance knobs (env-configurable) for faster completion.
    frame_stride = _env_int("VIDEO_ANALYSIS_FRAME_STRIDE", 2, min_value=1)
    target_detection_frames = _env_int("VIDEO_ANALYSIS_TARGET_DETECTION_FRAMES", 300, min_value=1)
    max_detection_dim = _env_int("VIDEO_ANALYSIS_MAX_DETECTION_DIM", 960, min_value=320)
    if total_frames > 0:
        auto_stride = max(1, int(total_frames / float(target_detection_frames)))
        frame_stride = max(frame_stride, auto_stride)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / f"video_asset_detection_{uuid.uuid4().hex}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise VisionEngineError("Unable to initialize output video writer.")

    aggregated_counts = {category: 0 for category in CATEGORY_ORDER}
    aggregated_area_px = {category: 0.0 for category in CATEGORY_ORDER}
    presence_frames = {category: 0 for category in CATEGORY_ORDER}
    processed_frames = 0
    detected_frames = 0
    frame_pixels = float(width * height)
    frame_index = 0
    last_counts: Dict[str, int] = {}
    last_area_pixels: Dict[str, float] = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            should_detect = processed_frames == 0 or ((frame_index - 1) % frame_stride == 0)

            if should_detect:
                detection_frame, scale = _prepare_detection_frame(frame, cv2, max_detection_dim)
                processed_small, frame_counts, frame_area_pixels, _ = _detect_assets_in_frame(
                    detection_frame,
                    cv2,
                    np,
                )
                detected_frames += 1

                if scale != 1.0:
                    processed_frame = cv2.resize(processed_small, (width, height), interpolation=cv2.INTER_LINEAR)
                    area_scale = 1.0 / (scale * scale)
                    frame_area_pixels = {
                        category: float(area_value) * area_scale
                        for category, area_value in frame_area_pixels.items()
                    }
                else:
                    processed_frame = processed_small

                last_counts = dict(frame_counts)
                last_area_pixels = dict(frame_area_pixels)
            else:
                frame_counts = {}
                frame_area_pixels = {}
                processed_frame = frame.copy()
                if last_counts or last_area_pixels:
                    _draw_info_panel(processed_frame, last_counts, last_area_pixels, int(frame_pixels), cv2)

            writer.write(processed_frame)
            processed_frames += 1

            for category in CATEGORY_ORDER:
                c = int(frame_counts.get(category, 0)) if should_detect else 0
                area_px = float(frame_area_pixels.get(category, 0.0)) if should_detect else 0.0
                aggregated_counts[category] += c
                aggregated_area_px[category] += area_px
                if c > 0 or area_px > 0:
                    presence_frames[category] += 1
    finally:
        cap.release()
        writer.release()

    if processed_frames == 0:
        if Path(output_path).exists():
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                pass
        raise VisionEngineError("No frames could be processed from uploaded video.")

    meters_per_pixel = get_meters_per_pixel()
    payload = default_response_schema()
    category_stats = payload["category_statistics"]

    detected_assets: List[Dict[str, Any]] = []
    total_events = 0
    category_coverage: Dict[str, float] = {}

    for category in CATEGORY_ORDER:
        event_count = int(aggregated_counts[category])
        avg_area_px = aggregated_area_px[category] / max(float(detected_frames), 1.0)
        coverage_percent = (avg_area_px / max(frame_pixels, 1.0)) * 100.0
        area_sq_m = pixel_area_to_sq_m(avg_area_px, meters_per_pixel)
        category_coverage[category] = max(0.0, min(100.0, coverage_percent))

        category_stats[category] = {
            "count": event_count,
            "total_area_sq_m": round(area_sq_m, 2),
            "coverage_percent": round(max(0.0, min(100.0, coverage_percent)), 4),
        }

        if event_count <= 0 and avg_area_px <= 0:
            continue

        presence_ratio = presence_frames[category] / max(float(detected_frames), 1.0)
        confidence = max(35.0, min(99.0, 45.0 + presence_ratio * 50.0))
        total_events += event_count
        detected_assets.append(
            {
                "unique_id": f"video_{category.lower().replace(' ', '_').replace('&', 'and')}",
                "category": category,
                "subcategory": "frame_level_detection",
                "confidence_percent": round(confidence, 2),
                "estimated_count": event_count,
                "estimated_area_sq_m": round(area_sq_m, 2),
                "estimated_dimensions_m": {"length": 0.0, "width": 0.0},
                "estimated_coverage_percent": round(max(0.0, min(100.0, coverage_percent)), 4),
                "condition_status": "Monitored",
                "maintenance_priority": "Medium",
                "center_coordinates": {"x": 50.0, "y": 50.0},
                "bounding_box": {"x_min": 5.0, "y_min": 5.0, "x_max": 95.0, "y_max": 95.0},
                "polygon_coordinates": [
                    {"x": 5.0, "y": 5.0},
                    {"x": 95.0, "y": 5.0},
                    {"x": 95.0, "y": 95.0},
                    {"x": 5.0, "y": 95.0},
                ],
                "visual_description": (
                    f"Frame-level detections for {category} across {detected_frames} analyzed frames."
                ),
                "estimation_basis": (
                    "OpenCV frame-by-frame segmentation; counts are detection events (not unique tracked IDs)."
                ),
            }
        )

    payload["detected_assets"] = detected_assets
    payload["image_analysis"] = {
        "scene_type": "Video Asset Map Analysis",
        "image_quality": "Processed",
        "estimated_total_area_sq_m": round(pixel_area_to_sq_m(frame_pixels, meters_per_pixel), 2),
        "overall_detection_confidence_percent": round(
            (sum(float(item["confidence_percent"]) for item in detected_assets) / max(len(detected_assets), 1)),
            2,
        ),
        "dominant_land_use": _dominant_land_use(category_coverage),
    }

    green_cover = category_coverage.get("Trees & Green Cover", 0.0) + category_coverage.get("Parks & Open Spaces", 0.0)
    built_up = category_coverage.get("Properties & Buildings", 0.0) + category_coverage.get("Vehicles & Parking", 0.0)
    road_pct = category_coverage.get("Roads & Footpaths", 0.0)
    water_pct = category_coverage.get("Water Bodies", 0.0)
    open_pct = category_coverage.get("Parks & Open Spaces", 0.0)

    payload["summary_statistics"] = {
        "total_assets_detected": total_events,
        "green_cover_percent": round(max(0.0, min(100.0, green_cover)), 2),
        "built_up_percent": round(max(0.0, min(100.0, built_up)), 2),
        "road_network_percent": round(max(0.0, min(100.0, road_pct)), 2),
        "water_body_percent": round(max(0.0, min(100.0, water_pct)), 2),
        "open_space_percent": round(max(0.0, min(100.0, open_pct)), 2),
    }

    payload["risk_analysis"] = {
        "encroachment_risk": "Moderate" if built_up > 45 else "Low",
        "drainage_risk": "Moderate" if aggregated_counts.get("Drains & Sewage", 0) > 0 else "Low",
        "environmental_risk": "Moderate" if green_cover < 20 else "Low",
        "flood_risk": "Moderate" if water_pct > 5 else "Low",
    }
    payload["ai_insights"] = [
        (
            f"Processed {processed_frames} frames at {fps:.2f} FPS source rate ({width}x{height}); "
            f"analyzed {detected_frames} frames with stride {frame_stride}."
        ),
        "Counts represent frame-level detection events from analyzed frames and are not de-duplicated object tracking IDs.",
        (
            f"Dominant land use trend is {payload['image_analysis']['dominant_land_use']} "
            f"with average coverage context from processed frames."
        ),
    ]

    validated, validation_warnings = validate_response_schema(payload)
    transformed = _build_transformed_payload(validated)

    warnings: List[str] = []
    warnings.extend(validation_warnings)
    warnings.append(
        (
            f"Video processing summary computed from {processed_frames} output frames; "
            f"{detected_frames} frames were fully analyzed (stride={frame_stride}, max_dim={max_detection_dim})."
        )
    )
    warnings.append(f"Area estimates use METERS_PER_PIXEL={meters_per_pixel}.")

    return AnalysisResult(
        validated_response=validated,
        transformed=transformed,
        raw_text=f"video_asset_map_processed_frames={processed_frames}",
        warnings=warnings,
        response_extras={
            "video_output_path": output_path,
            "video_metadata": {
                "processed_frames": processed_frames,
                "detected_frames": detected_frames,
                "source_total_frames": total_frames,
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "output_filename": os.path.basename(output_path),
                "frame_stride": frame_stride,
                "max_detection_dim": max_detection_dim,
            },
        },
    )

from __future__ import annotations

from math import radians, tan
from typing import Any, Dict, Tuple


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _line_box_intersection(
    center: Tuple[float, float],
    target: Tuple[float, float],
    bbox: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    """Find where the ray center->target exits the building rectangle."""
    x1, y1, x2, y2 = bbox
    cx, cy = center
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return cx, cy

    candidates = []

    if abs(dx) > 1e-6:
        t_left = (x1 - cx) / dx
        t_right = (x2 - cx) / dx
        candidates.extend([t_left, t_right])
    if abs(dy) > 1e-6:
        t_top = (y1 - cy) / dy
        t_bottom = (y2 - cy) / dy
        candidates.extend([t_top, t_bottom])

    valid_points = []
    for t in candidates:
        if t <= 0:
            continue
        px = cx + (t * dx)
        py = cy + (t * dy)
        if x1 - 1e-3 <= px <= x2 + 1e-3 and y1 - 1e-3 <= py <= y2 + 1e-3:
            valid_points.append((t, px, py))

    if not valid_points:
        return cx, cy

    valid_points.sort(key=lambda item: item[0])
    _, px, py = valid_points[0]
    return px, py


def compute_shadow_length(
    building_bbox: Tuple[int, int, int, int],
    shadow_contour: Any,
) -> Dict[str, Any] | None:
    """Compute longest shadow direction and length from building edge."""
    if shadow_contour is None:
        return None

    points = shadow_contour.reshape(-1, 2)
    if points.size == 0:
        return None

    center = _bbox_center(building_bbox)
    farthest_point = None
    farthest_distance = -1.0

    for p in points:
        px, py = float(p[0]), float(p[1])
        dist = _euclidean(center, (px, py))
        if dist > farthest_distance:
            farthest_distance = dist
            farthest_point = (px, py)

    if farthest_point is None:
        return None

    edge_point = _line_box_intersection(center, farthest_point, building_bbox)
    shadow_len_px = _euclidean(edge_point, farthest_point)

    return {
        "shadow_start_point": edge_point,
        "shadow_end_point": farthest_point,
        "shadow_length_pixels": float(max(shadow_len_px, 0.0)),
    }


def estimate_building_height(
    shadow_length_meters: float,
    solar_elevation_angle: float,
) -> float:
    """Estimate height from shadow geometry: H = S * tan(theta)."""
    safe_shadow = max(float(shadow_length_meters), 0.0)
    safe_angle = max(1e-3, min(float(solar_elevation_angle), 89.9))
    return safe_shadow * tan(radians(safe_angle))

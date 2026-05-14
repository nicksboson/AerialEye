from __future__ import annotations

from typing import Any, List, Sequence, Tuple


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _distance_point_to_bbox(point: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def _contour_centroid(contour: Any) -> Tuple[float, float]:
    import cv2  # type: ignore

    m = cv2.moments(contour)
    if m["m00"] == 0:
        p = contour.reshape(-1, 2)[0]
        return float(p[0]), float(p[1])
    cx = float(m["m10"] / m["m00"])
    cy = float(m["m01"] / m["m00"])
    return cx, cy


def _bbox_intersects_contour_bbox(
    bbox: Tuple[int, int, int, int],
    contour: Any,
    padding: int = 12,
) -> bool:
    import cv2  # type: ignore

    x1, y1, x2, y2 = bbox
    bx, by, bw, bh = cv2.boundingRect(contour)
    ax1, ay1, ax2, ay2 = x1 - padding, y1 - padding, x2 + padding, y2 + padding
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)


def detect_shadows(
    image_bgr: Any,
    min_shadow_area: float = 120.0,
    adaptive_block_size: int = 31,
    adaptive_c: int = 7,
) -> Tuple[Any, List[Any]]:
    """Detect candidate shadow regions from a single image.

    Steps:
    1. Grayscale conversion
    2. Gaussian smoothing
    3. Adaptive thresholding (dark region extraction)
    4. Morphological cleanup
    5. Contour filtering
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    block_size = max(3, int(adaptive_block_size) | 1)  # ensure odd
    threshold_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(adaptive_c),
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered: List[Any] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_shadow_area:
            continue
        filtered.append(contour)

    return cleaned, filtered


def associate_nearest_shadow(
    building_bbox: Tuple[int, int, int, int],
    shadow_contours: Sequence[Any],
    max_distance_factor: float = 2.6,
) -> Any | None:
    """Associate the nearest plausible shadow contour with a building bbox."""
    x1, y1, x2, y2 = building_bbox
    building_size = max(float(x2 - x1), float(y2 - y1), 1.0)
    max_distance = building_size * max_distance_factor
    cx, cy = _bbox_center(building_bbox)

    best_contour = None
    best_score = float("inf")
    for contour in shadow_contours:
        contour_center = _contour_centroid(contour)
        distance = _distance_point_to_bbox(contour_center, building_bbox)
        if distance > max_distance and not _bbox_intersects_contour_bbox(building_bbox, contour):
            continue

        # Favor contours touching/near the building; otherwise use center distance.
        touches_score = 0.0 if _bbox_intersects_contour_bbox(building_bbox, contour) else 15.0
        radial_distance = ((contour_center[0] - cx) ** 2 + (contour_center[1] - cy) ** 2) ** 0.5
        score = distance + touches_score + (0.05 * radial_distance)
        if score < best_score:
            best_score = score
            best_contour = contour

    return best_contour

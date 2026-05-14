from __future__ import annotations

from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt


def draw_height_overlay(
    analyzed_image_bgr: Any,
    building_height_results: Iterable[Dict[str, Any]],
) -> Any:
    """Draw building bbox, shadow contour, shadow line, and height label."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    output = analyzed_image_bgr.copy()

    for result in building_height_results:
        x1, y1, x2, y2 = result["bbox"]
        contour = result.get("shadow_contour")
        start_pt = result.get("shadow_start_point")
        end_pt = result.get("shadow_end_point")
        height_m = float(result.get("estimated_height_meters", 0.0))

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 80, 80), 2)

        if contour is not None and len(contour) > 0:
            cv2.drawContours(output, [contour], -1, (80, 255, 80), 2)

        if start_pt is not None and end_pt is not None:
            p1 = (int(round(start_pt[0])), int(round(start_pt[1])))
            p2 = (int(round(end_pt[0])), int(round(end_pt[1])))
            cv2.line(output, p1, p2, (80, 220, 255), 2)
            cv2.circle(output, p1, 3, (80, 220, 255), -1)
            cv2.circle(output, p2, 3, (80, 220, 255), -1)

        label = f"Height: {height_m:.1f} meters"
        label_y = max(14, y1 - 8)
        cv2.putText(
            output,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return output


def matplotlib_preview(image_bgr: Any, title: str = "Height Estimation Overlay") -> None:
    """Optional Matplotlib preview helper for Kaggle notebooks."""
    import cv2  # type: ignore

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

import os

DEFAULT_METERS_PER_PIXEL = 1.0


def get_meters_per_pixel() -> float:
    raw = os.getenv("METERS_PER_PIXEL", str(DEFAULT_METERS_PER_PIXEL)).strip()
    try:
        value = float(raw)
        if value <= 0:
            return DEFAULT_METERS_PER_PIXEL
        return value
    except (TypeError, ValueError):
        return DEFAULT_METERS_PER_PIXEL


def pixel_area_to_sq_m(pixel_area: float, meters_per_pixel: float | None = None) -> float:
    mpp = meters_per_pixel if meters_per_pixel is not None else get_meters_per_pixel()
    safe_area = max(float(pixel_area), 0.0)
    return safe_area * (mpp ** 2)


def pixel_length_to_m(pixel_length: float, meters_per_pixel: float | None = None) -> float:
    mpp = meters_per_pixel if meters_per_pixel is not None else get_meters_per_pixel()
    safe_length = max(float(pixel_length), 0.0)
    return safe_length * mpp

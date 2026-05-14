import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.services.vision_engine import VisionEngineError, analyze_spatial_image

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("spatialscan.backend")

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/webp",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="SpatialScan AI Analysis API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
if "*" in cors_origins and allow_credentials:
    logger.warning("CORS_ALLOW_CREDENTIALS=true is incompatible with wildcard origins. Forcing credentials off.")
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(request_id: str, code: str, message: str, details: str = "") -> Dict[str, Any]:
    return {
        "success": False,
        "request_id": request_id,
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "spatialscan-vision-analysis",
        "vision_api_key_loaded": bool(os.getenv("VISION_API_KEY")),
    }


@app.post("/api/analyze")
async def analyze_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    logger.info("Incoming analyze request id=%s filename=%s content_type=%s", request_id, image.filename, image.content_type)

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=_error_response(
                request_id,
                "invalid_image_type",
                "Unsupported image format. Use JPG, PNG, TIFF, or WEBP.",
                f"Received content type: {image.content_type}",
            ),
        )

    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail=_error_response(
                request_id,
                "empty_file",
                "Uploaded image is empty.",
            ),
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=_error_response(
                request_id,
                "file_too_large",
                "Uploaded file is too large.",
                f"Max allowed size: {MAX_UPLOAD_BYTES} bytes.",
            ),
        )

    suffix = Path(image.filename or "uploaded_image.jpg").suffix or ".jpg"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        result = analyze_spatial_image(temp_path)
        logger.info(
            "Analyze request complete id=%s assets=%s warnings=%s",
            request_id,
            len(result.validated_response.get("detected_assets", [])),
            len(result.warnings),
        )

        return {
            "success": True,
            "request_id": request_id,
            "data": result.validated_response,
            "transformed": result.transformed,
            "warnings": result.warnings,
        }
    except VisionEngineError as err:
        logger.exception("Vision analysis error id=%s", request_id)
        raise HTTPException(
            status_code=502,
            detail=_error_response(
                request_id,
                "vision_analysis_failed",
                "Vision AI analysis request failed.",
                str(err),
            ),
        ) from err
    except HTTPException:
        raise
    except Exception as err:
        logger.exception("Unexpected analysis failure id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=_error_response(
                request_id,
                "internal_error",
                "Unexpected backend failure during analysis.",
                str(err),
            ),
        ) from err
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove temp file: %s", temp_path)

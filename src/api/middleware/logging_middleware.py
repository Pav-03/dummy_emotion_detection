from src.utils.logger import get_logger
import time
import uuid
from fastapi import Request
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

logger = get_logger("api")

# path to kip logging ( avoid noisy)
SKIP_PATH = {"/health", "/docs", "/openapi.json", "/favicon.ico"}


async def log_request(request: Request, call_next):
    """Log every requet qith requet id , path, math, status and latency"""

    # skip noisy endpoint
    if request.url.path in SKIP_PATH:
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(uuid.uuid4())[:8]
        return response

    # Generate unique Request ID
    request_id = str(uuid.uuid4())[:8]

    # Before endpoint
    start_time = time.time()
    logger.info(f"[{request_id}] -> {request.method} {request.url.path}")

    # call the actual end point
    try:
        response = await call_next(request)

    except Exception as e:
        latency = time.time() - start_time
        logger.error(
            f"[{request_id}] * {request.method} {request.url.path} | Error: {e} | Latency: {latency:.3f}s")
        raise

    # AFTER endpoint
    latency = time.time() - start_time
    logger.info(
        f"[{request_id}] ← {request.method} {request.url.path} "
        f"| Status: {response.status_code} "
        f"| Latency: {latency:.3f}s"
    )

    # Add tracking headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(round(latency, 3))

    return response

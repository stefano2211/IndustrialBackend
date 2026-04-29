from typing import Any, Dict
import uuid
import time
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class GlobalExceptionHandler(BaseHTTPMiddleware):
    """Global exception handler middleware with request tracking."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"- {response.status_code} ({process_time:.1f}ms)"
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.1f}ms"

            return response

        except Exception as exc:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.exception(
                f"[{request_id}] {request.method} {request.url.path} "
                f"- 500 Error ({process_time:.1f}ms): {exc}"
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": f"{process_time:.1f}ms",
                },
            )
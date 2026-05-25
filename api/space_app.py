"""
Single FastAPI entrypoint for Hugging Face Docker Spaces.

Hugging Face exposes only one container port, so this app combines the
existing main API routes and ML engine routes under one server on port 7860.
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

_API_DIR = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_API_DIR, ".."))
for path in (_API_DIR, _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("CHROMA_DB_PATH", "/tmp/chroma")
os.makedirs(os.environ["CHROMA_DB_PATH"], exist_ok=True)

try:
    from .main import app as main_api
    from .engine import app as engine_api
except ImportError:
    from main import app as main_api
    from engine import app as engine_api


app = FastAPI(title="Indian News Comparator Space API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Indian News Comparator",
        "routes": ["/api/analyze", "/api/related", "/news", "/analyze_perspective"],
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def _include_api_routes(source_app: FastAPI, skip_paths: set[str] | None = None) -> None:
    skip_paths = skip_paths or set()
    for route in source_app.routes:
        if isinstance(route, APIRoute) and route.path not in skip_paths:
            app.router.routes.append(route)


_include_api_routes(main_api)
_include_api_routes(engine_api, skip_paths={"/api/ingest"})

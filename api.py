"""
Synchronous HTTP API for UniMate -- call it directly while testing.

Same request/response as the RunPod queue worker (both go through
``unimate.serving.service.run_motion``); here a ServiceError becomes an HTTP
status instead of a job error.

    uvicorn api:app --host 0.0.0.0 --port 8000

    GET  /health      -> 200 once the model is loaded, 503 before
    POST /v1/motion   -> {"motion": {...}, "meta": {...}}
                         400 bad_input / 422 bad_rig / 503 unavailable
"""

import os
import traceback
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from unimate.serving.engine import UniMateEngine
from unimate.serving.service import ServiceError, run_motion

EXP_DIR = os.environ.get("UNIMATE_EXP_DIR", "/opt/unimate/model")

_STATUS_BY_CODE = {"bad_input": 400, "bad_rig": 422, "unavailable": 503}

_engine = None
_load_error = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine, _load_error
    try:
        _engine = UniMateEngine(EXP_DIR)
    except Exception as e:  # noqa: BLE001 - /health and /v1/motion report it
        _load_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    yield


app = FastAPI(title="UniMate", lifespan=lifespan)


@app.get("/health")
def health():
    if _engine is None:
        raise HTTPException(503, detail=_load_error or "loading")
    return {"status": "ok", "device": str(_engine.device)}


@app.post("/v1/motion")
async def v1_motion(request: dict = Body(...)):
    if _engine is None:
        raise HTTPException(503, detail={"code": "unavailable", "error": _load_error or "loading"})
    try:
        # One GPU, one sampler: sampling in a worker thread keeps /health
        # responsive while a request runs.
        return await run_in_threadpool(run_motion, _engine, request)
    except ServiceError as e:
        raise HTTPException(_STATUS_BY_CODE.get(e.code, 500),
                            detail={"code": e.code, "error": str(e)})

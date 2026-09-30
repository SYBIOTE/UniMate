"""
RunPod queue-endpoint handler for UniMate.

Swaps the transport only: the request/response logic lives in
``unimate.serving.service.run_motion``, which the synchronous HTTP API (dev
branch) also calls, so the two deployments cannot drift apart.

The caller sends the skeleton it already has (a user upload or a SkinTokens
rig) and a prompt; the worker returns per-bone local rotations and root
positions for that same skeleton. Fields are documented in service.py.

    Job input:  {"op": "motion", "text": ..., "skeleton": {...}, "seed"?, "steps"?, ...}
    Job output: {"motion": {...}, "meta": {...}}
    On failure: {"error": "...", "code": "..."}

60 frames x 71 bones is ~150 KB, well under RunPod's ~2 MB /status cap, so
the result travels inline.
"""

import os
import traceback

import runpod

from unimate.serving.engine import UniMateEngine
from unimate.serving.service import ServiceError, run_motion

OP_MOTION = "motion"
EXP_DIR = os.environ.get("UNIMATE_EXP_DIR", "/opt/unimate/model")

_engine = None


def _error(code: str, message: str) -> dict:
    return {"error": str(message)[:2000], "code": code}


def handler(job: dict) -> dict:
    payload = job.get("input") or {}
    op = payload.get("op", OP_MOTION)
    if op != OP_MOTION:
        return _error("bad_op", f"Unsupported op {op!r}; expected {OP_MOTION!r}")
    if _engine is None:
        return _error("unavailable", "model failed to load at startup; see worker logs")
    try:
        return run_motion(_engine, payload)
    except ServiceError as e:
        return _error(e.code, e)
    except Exception as e:  # noqa: BLE001 - a dead worker tells the caller nothing
        print(traceback.format_exc(), flush=True)
        return _error("motion_failed", f"{type(e).__name__}: {e}")


# Load at import so the first job off the queue does not also pay for it.
print(">>> [queue] loading UniMate...", flush=True)
try:
    _engine = UniMateEngine(EXP_DIR)
    print(">>> [queue] runtime ready", flush=True)
except Exception as e:  # noqa: BLE001 - jobs report the real error
    print(f">>> [queue] [WARNING] runtime not loaded at startup: {e}", flush=True)
    traceback.print_exc()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

"""Request -> response for one motion job, shared by every transport.

The RunPod queue handler and the synchronous HTTP API both call
``run_motion``, so the two deployments cannot drift apart: they differ only
in how a ``ServiceError`` reaches the caller.

    request:  {"text": "...",
               "skeleton": {"names": [...], "parents": [...],
                            "positions": [[x,y,z]...], "quaternions": [[x,y,z,w]...],
                            "rootParentMatrix": [16 floats, column-major]?},
               "seed": int?, "cfg_scale": float?,
               "steps": int? (Euler steps, default 50; 0 = adaptive dopri5),
               "domain": "truebones" | "mixamo" | "objaverse"?,
               "labels": [...]?, "face_pair": [right, left]?, "body_axis": bool?}
    response: {"motion": {"fps", "frames", "bone_names",
                          "local_quat": [T][J][x,y,z,w], "root_pos": [T][x,y,z]},
               "meta": {"labels", "face_pair", "domain", "steps", "seed", "ms"}}
"""

import time

from unimate.serving.engine import DOMAINS, UniMateEngine
from unimate.serving.rig import RigError

DEFAULT_STEPS = 50


class ServiceError(Exception):
    """A job the caller got wrong. ``code`` is stable; branch on it, not the text."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def run_motion(engine: UniMateEngine, request: dict) -> dict:
    skeleton = request.get("skeleton")
    text = request.get("text")
    if not isinstance(skeleton, dict) or not isinstance(text, str) or not text.strip():
        raise ServiceError("bad_input", "'skeleton' (object) and 'text' (non-empty string) are required")
    domain = request.get("domain", "truebones")
    if domain not in DOMAINS:
        raise ServiceError("bad_input", f"'domain' must be one of {DOMAINS}")
    steps = int(request.get("steps", DEFAULT_STEPS))
    seed = request.get("seed")

    started = time.time()
    try:
        out = engine.generate(
            skeleton, text.strip(),
            seed=int(seed) if seed is not None else None,
            cfg_scale=request.get("cfg_scale"),
            domain=domain,
            labels=request.get("labels"),
            face_pair=request.get("face_pair"),
            body_axis=bool(request.get("body_axis", False)),
            steps=steps or None,
        )
    except RigError as e:
        raise ServiceError("bad_rig", str(e)) from e
    except (KeyError, ValueError) as e:
        raise ServiceError("bad_input", str(e)) from e

    return {
        "motion": {
            "fps": 30,
            "frames": int(out.local_quat.shape[0]),
            "bone_names": list(skeleton["names"]),
            "local_quat": out.local_quat.round(5).tolist(),
            "root_pos": out.root_pos.round(5).tolist(),
        },
        "meta": {
            "labels": list(out.labels),
            "face_pair": list(out.face_pair) if out.face_pair else None,
            "domain": out.domain,
            "steps": steps,
            "seed": seed,
            "ms": int((time.time() - started) * 1000),
        },
    }

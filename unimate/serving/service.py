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
               "loop": true | false | "auto"? (default auto: cyclic prompts loop),
               "foot_lock": bool? (default true: pin feet during contacts),
               "domain": "truebones" | "mixamo" | "objaverse"?,
               "labels": [...]?, "face_pair": [right, left]?, "body_axis": bool?}
    response: {"motion": {"fps", "frames", "bone_names",
                          "local_quat": [T][J][x,y,z,w], "root_pos": [T][x,y,z],
                          "loop", "root_velocity": [x,y,z] | null},
               "meta": {"labels", "face_pair", "body_axis", "body_plan", "domain",
                        "steps", "seed", "ms"}}
"""

import re
import time

from unimate.serving.engine import DOMAINS, UniMateEngine
from unimate.serving.rig import RigError

DEFAULT_STEPS = 50
# Prompts for cyclic motion loop by default ("loop": "auto").
CYCLIC = re.compile(r'\b(walk|run|trot|gallop|canter|jog|sprint|crawl|slither|swim|fly|flies|flying|flap|hop|march|prowl|pace|idle|breath)', re.IGNORECASE)


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
    loop = request.get("loop", "auto")
    if loop not in (True, False, "auto"):
        raise ServiceError("bad_input", "'loop' must be true, false or \"auto\"")
    if loop == "auto":
        loop = bool(CYCLIC.search(text))

    started = time.time()
    try:
        out = engine.generate(
            skeleton, text.strip(),
            seed=int(seed) if seed is not None else None,
            cfg_scale=request.get("cfg_scale"),
            domain=domain,
            labels=request.get("labels"),
            face_pair=request.get("face_pair"),
            body_axis=request.get("body_axis"),
            steps=steps or None,
            loop=loop,
            foot_lock=bool(request.get("foot_lock", True)),
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
            # Loops: the last frame closes the cycle and the root keeps its
            # travel (root motion), ending one cycle ahead of where it started.
            # root_velocity is that travel's average per second.
            "loop": out.loop,
            "root_velocity": out.root_velocity.round(5).tolist() if out.root_velocity is not None else None,
        },
        "meta": {
            "labels": list(out.labels),
            "face_pair": list(out.face_pair) if out.face_pair else None,
            "body_axis": out.body_axis,
            "body_plan": out.plan,
            "domain": out.domain,
            "steps": steps,
            "seed": seed,
            "ms": int((time.time() - started) * 1000),
        },
    }

# RunPod — UniMate queue worker

Text-to-motion for **the caller's own skeleton**: a user upload or a SkinTokens
rig, humanoid or not. The job carries the rest pose; the result is per-bone
local rotations plus root positions for those same bones. No dataset is needed
at runtime — `unimate/serving/rig.py` builds the skeleton condition the
released `sample.py` would read from a stage-4 `cond.npy`.

## Layout

| File | Role |
|------|------|
| `Dockerfile.base` | CUDA 12.4 + Python 3.10 + `requirements-serve.txt` + weights (EMA-only v2 checkpoint, flan-t5-base), pinned HF revisions |
| `Dockerfile` | Queue worker on top of the base: `runpod`, current `unimate/`, `data_process/`, `handler.py` |
| `handler.py` | Job in/out, error codes |
| `unimate/serving/rig.py` | Caller rig ⇄ canonical skeleton (stage-4 canonicalization and its inverse) |
| `unimate/serving/names.py` | Joint labels the model conditions on; facing pair |
| `unimate/serving/engine.py` | Loads once, builds the one-sample batch, samples |
| `scripts/slim_checkpoint.py` | 1.2 GB training checkpoint → 300 MB EMA weights |

Weights are baked into the base (1.3 GB total), not on the network volume, so
the endpoint is not tied to a datacenter.

## Build and push

```bash
docker build -f Dockerfile.base -t sybiote/unimate-base:latest .
docker push sybiote/unimate-base:latest

docker build --build-arg BASE_IMAGE=sybiote/unimate-base:latest -t sybiote/unimate-queue:<tag> .
docker push sybiote/unimate-queue:<tag>
```

Code-only changes need just the second pair (or a RunPod Git build of
`Dockerfile` on this branch). Rebuild the base only for dependency or weight
changes.

## Endpoint

Queue endpoint, AMPERE_16, 1 GPU, `minCudaVersion` 12.4, 0 min workers,
FlashBoot. No env vars, no volume. Warm inference is a few seconds at the
default 50 steps; cold start is the image pull plus ~10 s model load.

## Job contract

```jsonc
// input
{
  "op": "motion",
  "text": "An object trots forward.",
  "skeleton": {
    "names": ["pelvis", "spine", ...],          // unique
    "parents": [-1, 0, ...],                    // single root, at index 0
    "positions": [[x, y, z], ...],              // rest bone.position
    "quaternions": [[x, y, z, w], ...],         // rest bone.quaternion
    "rootParentMatrix": [16 floats]             // optional: root's parent matrixWorld.elements
  },
  "seed": 3,             // optional
  "steps": 50,           // optional: Euler steps; 0 = adaptive dopri5 (paper default, ~4x slower)
  "cfg_scale": 3.0,      // optional: defaults to the checkpoint's value
  "domain": "truebones", // optional: normalization stats — truebones | mixamo | objaverse
  "labels": [...],       // optional: override the joint labels
  "face_pair": ["R thigh", "L thigh"], // optional: override the facing pair (bone names)
  "body_axis": false     // optional: true for snakes (face_pair = [head, tail])
}
// output
{
  "motion": {"fps": 30, "frames": 60, "bone_names": [...],
             "local_quat": [T][J][x, y, z, w],   // set bone.quaternion
             "root_pos":   [T][x, y, z]},        // set root bone.position
  "meta": {"labels": [...], "face_pair": [...], "domain": "...", "steps": 50, "seed": 3, "ms": 6800}
}
// failure
{"error": "...", "code": "bad_input" | "bad_rig" | "bad_op" | "unavailable" | "motion_failed"}
```

Rig requirements: Y-up world, uniform scale on the root's parent, ≤ 71 bones,
hierarchy depth ≤ 19 (`bad_rig` otherwise). The motion starts at the rig's
rest position, facing the rig's own forward.

### Labels

The model embeds each joint's cleaned name with T5, so names are conditioning.
Named rigs (Mixamo, Bip01, Rigify, `thigh.L`, …) go through the authors' rule
cleaner. Placeholder names (`bone_12`, as SkinTokens emits for non-humanoids)
are labelled from rest-pose geometry instead: mirror symmetry gives left/right,
feet give forward, and chains are classed as Leg/Foot/Arm/Hand/Tail/Spine/Neck/Head.
`meta.labels` returns what was used.

### Prompts

Training captions use a generic subject ("An object trots forward."), since the
same model covers every species. Prompts phrased that way match the training
distribution.

## Test

```bash
curl -sS -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d @job.json
```

Locally, without Docker: `UNIMATE_EXP_DIR=<model dir> python handler.py --test_input "$(cat job.json)"`.

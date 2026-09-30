# UniMate — synchronous HTTP API (testing)

`api.py` serves the same request/response as the RunPod queue worker
(`runpod-queue` branch) over plain HTTP, so a rig can be tried directly
without the queue. Both call `unimate.serving.service.run_motion`.

## Run

Local (needs a model dir: `config.json`, `dataset_stats.npy`, `checkpoints/`):

```bash
pip install "setuptools<81" && pip install --no-build-isolation -r requirements-serve.txt -r requirements-api.txt
python scripts/slim_checkpoint.py <hf exp dir> <hf ckpt.pt> <model dir>/checkpoints/checkpoint_step_100000.pt
UNIMATE_EXP_DIR=<model dir> uvicorn api:app --port 8000
```

Docker, like the local Kimodo container (weights mounted from `./unimate-weights`, port 8010):

```bash
make build-base      # deps only; rebuild when requirements change
make build           # the API image
make weights         # once: pinned checkpoint (EMA-only) + flan-t5-base, ~1.3 GB
make dev             # docker compose up; curl localhost:8010/health
```

Point the app at it with `DEV_UNIMATE_API_URL=http://localhost:8010` in
`nirvana-animate-saas/.env.local`.

## Endpoints

| | |
|---|---|
| `GET /health` | 200 `{"status": "ok"}` once loaded; 503 while loading or if loading failed |
| `POST /v1/motion` | 200 `{"motion", "meta"}`; 400 `bad_input`, 422 `bad_rig`, 503 `unavailable` — body `{"detail": {"code", "error"}}` |

```bash
curl -s -X POST localhost:8000/v1/motion -H 'Content-Type: application/json' -d @request.json
```

Request and response fields are documented in `unimate/serving/service.py`.
In short: send the rig's rest pose (`names`, `parents`, `positions`,
`quaternions` as x,y,z,w — Three.js `bone.position` / `bone.quaternion`), a
`text` prompt, and optionally `seed`, `steps` (default 50), `domain`,
`labels`, `face_pair`. Back come 60 frames at 30 fps of `local_quat` per bone
and `root_pos` for the root, in the same bone order.

Rig requirements: single root at index 0, Y-up, ≤ 71 bones, depth ≤ 19.
Prompts match training best phrased with a generic subject:
"An object trots forward."

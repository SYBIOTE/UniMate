#!/usr/bin/env bash
# Resolve UniMate weights, then exec the command (the queue handler or uvicorn).
#
# Weights live outside the image, as for Kimodo / HY-Motion: a mounted
# ./unimate-weights locally, the network volume on RunPod. The root is
# UNIMATE_WEIGHTS_ROOT, else /runpod-volume/unimate when it exists. Seed it
# once with `python scripts/seed_weights.py <root>`.
#
# With no root, nothing is exported and UNIMATE_EXP_DIR must already point at a
# model dir (plain `uvicorn` against a local checkout).
set -euo pipefail

ROOT="${UNIMATE_WEIGHTS_ROOT:-}"
if [[ -z "${ROOT}" && -d /runpod-volume/unimate ]]; then
  ROOT=/runpod-volume/unimate
fi

if [[ -n "${ROOT}" ]]; then
  MODEL="${UNIMATE_MODEL:-unimate_uniml3d_f60_v2}"
  export UNIMATE_EXP_DIR="${ROOT}/model/${MODEL}"
  export HF_HUB_CACHE="${ROOT}/hf"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  T5="${HF_HUB_CACHE}/models--google--flan-t5-base"

  missing=()
  for f in "${UNIMATE_EXP_DIR}/config.json" "${UNIMATE_EXP_DIR}/dataset_stats.npy" \
           "${T5}/refs/main"; do
    [[ -f "${f}" ]] || missing+=("${f}")
  done
  compgen -G "${UNIMATE_EXP_DIR}/checkpoints/checkpoint_step_*.pt" >/dev/null \
    || missing+=("${UNIMATE_EXP_DIR}/checkpoints/checkpoint_step_*.pt")
  compgen -G "${T5}/snapshots/*/model.safetensors" >/dev/null \
    || missing+=("${T5}/snapshots/*/model.safetensors")
  if (( ${#missing[@]} )); then
    echo "ERROR: UniMate weights incomplete under ${ROOT}:" >&2
    printf '  missing %s\n' "${missing[@]}" >&2
    echo "Seed them with: python scripts/seed_weights.py ${ROOT}" >&2
    exit 1
  fi
  echo ">>> UniMate weights: ${ROOT} (model ${MODEL})"
fi

exec "$@"

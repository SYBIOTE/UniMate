"""Seed a UniMate weights root: local ./unimate-weights or the RunPod volume.

Layout (what docker-entrypoint.sh expects):

    {root}/model/{MODEL}/config.json
    {root}/model/{MODEL}/dataset_stats.npy
    {root}/model/{MODEL}/checkpoints/checkpoint_step_{STEP}.pt   (EMA-only, ~300 MB)
    {root}/hf/models--google--flan-t5-base/...    (a Hugging Face hub cache: HF_HUB_CACHE)

UniMate's T5 wrapper takes the model *name* (it sizes the embedding from it),
so the encoder stays a hub cache the name resolves in offline. An exact-commit
download writes the snapshot but no refs/main, and offline resolution of
"google/flan-t5-base" goes through refs/main -- so it is written here.

Revisions are pinned; the model card notes checkpoints keep moving. Idempotent:
files already present are kept.

    python scripts/seed_weights.py ./unimate-weights
    python scripts/seed_weights.py /runpod-volume/unimate
"""

import os
import sys
import tempfile

# Runnable as `python scripts/seed_weights.py` from a checkout or the image:
# slim_checkpoint imports the unimate package from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download, snapshot_download

UNIMATE_REPO = 'Linzhan/UniMate'
UNIMATE_REV = os.environ.get('UNIMATE_REV', '902b8465567159a974de2d10f129e29787b9e4e4')
MODEL = os.environ.get('UNIMATE_MODEL', 'unimate_uniml3d_f60_v2')
STEP = os.environ.get('UNIMATE_STEP', '100000')
T5_REPO = 'google/flan-t5-base'
T5_REV = '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'
T5_FILES = ('config.json', 'model.safetensors', 'spiece.model', 'tokenizer.json',
            'tokenizer_config.json', 'special_tokens_map.json')


def seed(root: str) -> None:
    model_dir = os.path.join(root, 'model', MODEL)
    ckpt = os.path.join(model_dir, 'checkpoints', f'checkpoint_step_{STEP}.pt')
    hub_cache = os.path.join(root, 'hf')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    for name in ('config.json', 'dataset_stats.npy'):
        if not os.path.exists(os.path.join(model_dir, name)):
            print(f'>>> {UNIMATE_REPO}/{MODEL}/{name}', flush=True)
            hf_hub_download(UNIMATE_REPO, f'{MODEL}/{name}', revision=UNIMATE_REV, local_dir=os.path.join(root, 'model'))

    if not os.path.exists(ckpt):
        # The released file carries optimizer state (1.2 GB); keep only the EMA weights.
        from scripts.slim_checkpoint import main as slim
        with tempfile.TemporaryDirectory() as tmp:
            print(f'>>> {UNIMATE_REPO}/{MODEL} step {STEP} (1.2 GB, slimmed to EMA)', flush=True)
            src = hf_hub_download(UNIMATE_REPO, f'{MODEL}/checkpoints/checkpoint_step_{STEP}.pt',
                                  revision=UNIMATE_REV, local_dir=tmp)
            slim(model_dir, src, ckpt)

    print(f'>>> {T5_REPO} @ {T5_REV[:8]}', flush=True)
    snapshot_download(T5_REPO, revision=T5_REV, cache_dir=hub_cache, allow_patterns=list(T5_FILES))
    refs = os.path.join(hub_cache, 'models--' + T5_REPO.replace('/', '--'), 'refs')
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, 'main'), 'w') as f:
        f.write(T5_REV)

    print(f'>>> UniMate weights ready under {root}', flush=True)


if __name__ == '__main__':
    seed(sys.argv[1] if len(sys.argv) > 1 else '/runpod-volume/unimate')

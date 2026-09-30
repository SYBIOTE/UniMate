"""Bake EMA weights into a released checkpoint and drop optimizer state.

A released ``checkpoint_step_*.pt`` carries model, EMA, optimizer and
scheduler state (~1.2 GB). Inference only ever reads the EMA copy, so the
serving image keeps just that, written as ``model_state_dict`` (~300 MB).
``_load_checkpoint`` falls back to the plain weights when no EMA entry
exists, so the slim file loads through the same code path.

    python scripts/slim_checkpoint.py <exp_dir> <src.pt> <dst.pt>
"""

import os
import sys

import torch

from unimate.configs.schema import MainConfig
from unimate.inference.sample import _load_checkpoint
from unimate.models.factory import create_model


def main(exp_dir: str, src: str, dst: str):
    config = MainConfig.from_json(os.path.join(exp_dir, 'config.json'))
    model = create_model(dataset_config=config.dataset, model_config=config.model)
    _load_checkpoint(model, src, config)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, dst)
    print(f'{src} ({os.path.getsize(src) / 1e6:.0f} MB) -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main(*sys.argv[1:4])

"""Load UniMate once; generate motion for a caller rig and a prompt.

Mirrors ``unimate.inference.sample`` + ``create_sample_condition`` with the
dataset taken out: the skeleton comes from ``rig.prepare_rig`` instead of a
stage-4 ``cond.npy``, and only the normalization stats are read from disk
(they ship beside the checkpoint).
"""

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from Animation import offsets_from_positions
from Quaternions import Quaternions

from unimate.configs.schema import MainConfig
from unimate.dataset.mixture.collate import mixture_batch_collate
from unimate.dataset.mixture.dataset import create_condition
from unimate.dataset.transforms import apply_normalization, build_parent_features
from unimate.inference.generate import ClassifierFreeSampleModel, generate_samples
from unimate.inference.sample import _build_diffusion, _encode_prompt, _load_checkpoint
from unimate.models.factory import create_model
from unimate.models.text_encoder.factory import create_text_encoder
from unimate.serving import names as rig_names
from unimate.serving.rig import RigInput, prepare_rig, features_to_rig
from unimate.utils.text_emb_cache import pool, sequences_from_hidden

DOMAINS = ('truebones', 'mixamo', 'objaverse')
FPS = 30


@dataclass
class Generated:
    local_quat: np.ndarray       # (T, J, 4) xyzw, caller joint order
    root_pos: np.ndarray         # (T, 3) root parent space
    labels: Sequence[str]        # cleaned names the model was conditioned on
    face_pair: Optional[Sequence[str]]
    domain: str


class UniMateEngine:
    def __init__(self, exp_dir: str, checkpoint: Optional[str] = None,
                 device: Optional[str] = None):
        self.config = MainConfig.from_json(os.path.join(exp_dir, 'config.json'))
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ds = self.config.dataset
        self.max_joints, self.max_depth = ds.max_joints, ds.max_depth
        self.T = ds.max_motion_length
        self.stats = np.load(os.path.join(exp_dir, 'dataset_stats.npy'), allow_pickle=True).item()

        self.model = create_model(dataset_config=ds, model_config=self.config.model)
        _load_checkpoint(self.model, checkpoint or _latest_checkpoint(exp_dir), self.config)
        self.model.to(self.device).eval()
        self.diffusion, self.gen_diffusion = _build_diffusion(self.config)

        self.text_encoder = create_text_encoder(
            encoder_type=self.config.model.text_encoder_type,
            encoder_version=self.config.model.text_encoder_version,
            device=str(self.device), pool=False,
        )
        self._name_emb: Dict[str, np.ndarray] = {}

    # -- text -------------------------------------------------------------

    def _embed_names(self, labels: Sequence[str]) -> np.ndarray:
        todo = sorted({n for n in labels if n not in self._name_emb})
        if todo:
            with torch.no_grad():
                inputs = self.text_encoder.tokenize(todo)
                hidden = self.text_encoder(inputs).detach().cpu()
            seqs = sequences_from_hidden(hidden, inputs['attention_mask'].cpu())
            for n, s in zip(todo, seqs):
                self._name_emb[n] = pool(s)
        return np.stack([self._name_emb[n] for n in labels])

    # -- batch ------------------------------------------------------------

    def _batch(self, cond_entry: Dict, domain: str, caption: str) -> Dict:
        """One-sample batch, built as ``create_sample_condition`` does."""
        condition = create_condition(cond_entry, max_freqs=self.config.model.max_freqs)
        if self.config.dataset.ground_motion_height:
            condition['tpos'] = condition['tpos'].copy()
            condition['tpos'][..., 1] -= condition['tpos'][..., 1].min()
        parents = condition['parents']
        J, D = len(parents), self.config.dataset.feature_len

        tpos = np.zeros((J, D))
        tpos[:, :3] = condition['tpos']
        tpos[:, 3:9] = Quaternions.id(1).rotation_matrix(cont6d=True)[0]

        s = self.stats[domain]
        mean, std = np.zeros((J, D)), np.ones((J, D))
        mean[0], std[0] = s['mean_root'], s['std_root']
        mean[1:], std[1:] = s['mean_local'], s['std_local']
        tpos_n = apply_normalization(tpos, mean, std)

        batch = {
            # The sampler starts from noise; this only fixes shapes.
            'motion': np.zeros((self.T, J, D), dtype=np.float32),
            'max_motion_length': self.T,
            'motion_length': self.T,
            'max_joints': self.max_joints,
            'parents': parents,
            'edge_indexs': condition['edge_indexs'],
            'tpos_first_frame': tpos_n,
            'tpos_first_frame_parents': build_parent_features(tpos_n, parents)['tpos_first_frame_parents'],
            'offsets': offsets_from_positions(condition['tpos'], parents),
            'joint_graph_dist': condition['joint_graph_dist'],
            'joint_relations': condition['joint_relations'],
            'joint_depths': condition['joint_depths'],
            'spectral_feats': condition['spectral_feats'],
            'joint_names_emb': self._embed_names(cond_entry['clean_joint_names']),
            'object_type': 'rig',
            'start_idx': 0,
            'mean': mean,
            'std': std,
            'split_tag': 'train',
            'caption': caption,
        }
        batch.update(_encode_prompt(self.text_encoder, caption))
        return batch

    # -- public -----------------------------------------------------------

    @torch.no_grad()
    def generate(self, skeleton: Dict, text: str, *, seed: Optional[int] = None,
                 cfg_scale: Optional[float] = None, domain: str = 'truebones',
                 labels: Optional[Sequence[str]] = None,
                 face_pair: Optional[Sequence[str]] = None,
                 body_axis: bool = False, steps: Optional[int] = None) -> Generated:
        if domain not in DOMAINS:
            raise ValueError(f'domain must be one of {DOMAINS}')
        rig = RigInput.from_payload(skeleton)
        labels = list(labels) if labels else self._auto_labels(rig)
        if len(labels) != len(rig.names):
            raise ValueError('labels must have one entry per bone')
        if face_pair is None:
            face_pair = rig_names.pick_face_pair(rig.names, labels, rig.parents, _depths(rig.parents))

        ctx = prepare_rig(rig, labels, face_pair=face_pair, body_axis=body_axis,
                          max_joints=self.max_joints, max_depth=self.max_depth,
                          max_freqs=self.config.model.max_freqs)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        _, cond = mixture_batch_collate([self._batch(ctx.cond, domain, text)])
        cond = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in cond.items()}

        shape = (1, self.max_joints, self.config.dataset.feature_len, self.T)
        cfg = cfg_scale if cfg_scale is not None else self.config.sampling.cfg_scale
        if steps:
            # Fixed-step Euler: the solver the authors' constraint sampler uses.
            # Default (None) keeps the paper's adaptive dopri5.
            model = ClassifierFreeSampleModel(self.model, cfg) if cfg > 1.0 else self.model
            noise = torch.randn(shape, device=self.device)
            samples = self.gen_diffusion.sample_ode(sampling_method='euler', num_steps=steps)(
                noise, model, cond=cond)[-1]
        else:
            samples = generate_samples(
                model=self.model, cond=cond, motion_shape=shape,
                diff_model=self.config.training.diff_model, diffusion=self.diffusion,
                gen_diffusion=self.gen_diffusion, device=self.device, cfg_scale=cfg,
            )

        J = len(rig.names)
        feats = samples[0, :J].cpu().permute(2, 0, 1).numpy().astype(np.float64)  # (T, J, D)
        feats = feats * cond['std'][0][:J].cpu().numpy()[None] + cond['mean'][0][:J].cpu().numpy()[None]
        local_quat, root_pos = features_to_rig(feats, ctx)
        return Generated(local_quat, root_pos, labels, face_pair, domain)

    def _auto_labels(self, rig: RigInput):
        if not rig_names.looks_generic(rig.names):
            return rig_names.clean_names(rig.names)
        return rig_names.geometric_labels(rig.parents, _rest_world_positions(rig))


def _latest_checkpoint(exp_dir: str) -> str:
    paths = glob.glob(os.path.join(exp_dir, 'checkpoints', 'checkpoint_step_*.pt'))
    if not paths:
        raise FileNotFoundError(f'no checkpoints in {exp_dir}/checkpoints')
    return max(paths, key=lambda p: int(re.search(r'(\d+)\.pt$', p).group(1)))


def _depths(parents: np.ndarray) -> np.ndarray:
    d = np.zeros(len(parents), dtype=int)
    for j in range(len(parents)):
        p = parents[j]
        while p >= 0:
            d[j] += 1
            p = parents[p]
    return d


def _rest_world_positions(rig: RigInput) -> np.ndarray:
    """Rest-pose world positions for labeling (Y-up, +Z forward assumed)."""
    rot = rig.local_quat[:, [3, 0, 1, 2]]
    g_rot = [None] * len(rig.parents)
    g_pos = np.zeros((len(rig.parents), 3))
    m = rig.root_parent
    s = np.linalg.norm(m[:3, :3], axis=0).mean()
    # Parents before children, whatever order the caller listed bones in.
    order = np.argsort(_depths(rig.parents), kind='stable')
    for j in order:
        p = rig.parents[j]
        if p < 0:
            g_pos[j] = (m @ np.append(rig.local_pos[j], 1.0))[:3]
            g_rot[j] = Quaternions.from_transforms((m[:3, :3] / s)[None]) * Quaternions(rot[j:j + 1])
        else:
            g_pos[j] = g_pos[p] + (g_rot[p] * (rig.local_pos[j:j + 1] * s))[0]
            g_rot[j] = g_rot[p] * Quaternions(rot[j:j + 1])
    return g_pos

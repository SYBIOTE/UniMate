"""Caller rig <-> UniMate canonical skeleton.

The released inference path reads each target skeleton from a stage-4
``cond.npy``. Everything in that entry is derived from the rest pose alone
(no motion), so this module rebuilds it from a rig the caller sends, and
maps generated features back onto that rig's own bones.

Why the mapping back is exact regardless of the rig's bone-axis convention
(Blender, glTF, Maya): the model's rotation channels are T-pose-rebased
(``compute_rots_from_tpos``). With ``C`` the rest global rotations and ``D``
each joint's world-space delta from rest, the rebased rotation of joint j is
``D_parent^-1 * D_j`` -- the per-bone local frame cancels. So we recover
``D`` by chaining the rebased rotations, then rebuild local rotations as
``G_parent^-1 * G_j`` with ``G = D * C`` in the caller's own frames.

Input rig convention (matches Three.js ``Bone``):
    names:          (J,) bone names, unique
    parents:        (J,) parent index, -1 for the single root
    local_pos:      (J, 3) rest ``bone.position`` (parent space)
    local_quat:     (J, 4) rest ``bone.quaternion`` as x, y, z, w
    root_parent:    (4, 4) world matrix of the root's parent (row-major).
                    Must map into a Y-up space. Uniform scale only.

Output (``features_to_rig``), in the caller's joint order:
    local_quat:     (T, J, 4) x, y, z, w -- set ``bone.quaternion``
    root_pos:       (T, 3) -- set the root ``bone.position`` (parent space)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from Animation import positions_global, rotations_global
from Quaternions import Quaternions

from data_process.utils.motion_features import (
    MIN_SKELETON_DIAMETER,
    _load_tpos_npz,
    build_topology_cond,
    resolve_face_joint_idxs,
)
from data_process.utils.skeleton import (
    canonicalize_anim,
    get_root_facing_quat,
    get_skeleton_diameter,
)
from Animation import offsets_from_positions
from unimate.utils.motion_utils import (
    hml_rotations_to_bvh_quaternions,
    recover_unimate_joint_pos_from_rot,
    recover_unimate_root_quat_and_pos,
)

TARGET_DIAMETER = 2.0


class RigError(ValueError):
    """The rig cannot be animated as sent (bad topology, too large, ...)."""


@dataclass
class RigInput:
    names: List[str]
    parents: np.ndarray          # (J,)
    local_pos: np.ndarray        # (J, 3)
    local_quat: np.ndarray       # (J, 4) xyzw
    root_parent: np.ndarray      # (4, 4)

    @classmethod
    def from_payload(cls, skel: Dict) -> 'RigInput':
        names = [str(n) for n in skel['names']]
        parents = np.asarray(skel['parents'], dtype=np.int64)
        local_pos = np.asarray(skel['positions'], dtype=np.float64).reshape(-1, 3)
        local_quat = np.asarray(skel['quaternions'], dtype=np.float64).reshape(-1, 4)
        m = skel.get('rootParentMatrix')
        # Three.js Matrix4.elements is column-major; accept it as sent.
        root_parent = (np.asarray(m, dtype=np.float64).reshape(4, 4).T
                       if m is not None else np.eye(4))
        return cls(names, parents, local_pos, local_quat, root_parent)


@dataclass
class RigContext:
    """Everything needed to condition the model and to map its output back."""
    cond: Dict                   # stage-4-compatible cond entry (BFS order)
    bfs_order: List[int]         # bfs index k -> caller joint bfs_order[k]
    parents_bfs: np.ndarray      # (J,)
    facing: Quaternions          # (1,) canonical = facing * world
    xz: np.ndarray               # (3,) XZ subtracted after facing
    ground: float                # Y subtracted after scaling
    scale: float                 # canonical = scale * world
    rest_world_rot: Quaternions  # (J,) caller rest global rotations, BFS order
    parent_rot: Quaternions      # (1,) rotation part of root_parent
    root_parent: np.ndarray      # (4, 4)


def _wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return q_xyzw[..., [3, 0, 1, 2]]


def _xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return q_wxyz[..., [1, 2, 3, 0]]


def _decompose_uniform(m: np.ndarray):
    lin = m[:3, :3]
    scales = np.linalg.norm(lin, axis=0)
    s = float(scales.mean())
    if s <= 0 or np.ptp(scales) > 1e-3 * s:
        raise RigError(f'rootParentMatrix must have uniform positive scale, got {scales}')
    return s, Quaternions.from_transforms((lin / s)[None])


def _validate_topology(parents: np.ndarray, names: Sequence[str],
                       max_joints: int, max_depth: int):
    J = len(parents)
    if len(set(names)) != J:
        raise RigError('bone names must be unique')
    roots = np.flatnonzero(parents < 0)
    if len(roots) != 1:
        raise RigError(f'expected exactly one root bone, got {len(roots)}')
    if J < 2:
        raise RigError('skeleton needs at least 2 bones')
    if J > max_joints:
        raise RigError(f'{J} bones exceeds the model limit of {max_joints}')
    depth = np.zeros(J, dtype=int)
    for j in range(J):
        d, p, seen = 0, parents[j], 0
        while p >= 0:
            d, p, seen = d + 1, parents[p], seen + 1
            if seen > J:
                raise RigError('parent indices contain a cycle')
        depth[j] = d
    if depth.max() > max_depth:
        raise RigError(f'hierarchy depth {depth.max()} exceeds the model limit of {max_depth}')


def prepare_rig(rig: RigInput, clean_names: Sequence[str],
                face_pair: Optional[Sequence[str]] = None, body_axis: bool = False,
                max_joints: int = 71, max_depth: int = 19,
                max_freqs: int = 8) -> RigContext:
    """Canonicalize the caller's rest pose exactly as stage 4 does."""
    _validate_topology(rig.parents, rig.names, max_joints, max_depth)
    root = int(np.flatnonzero(rig.parents < 0)[0])
    if root != 0:
        # Stage 4 and the Animation lib assume joint 0 is the root.
        raise RigError('root bone must be first (index 0)')

    s, parent_rot = _decompose_uniform(rig.root_parent)

    # Express the rest pose in world units: the root in world space, children
    # scaled by the parent's uniform scale (their rotations are unchanged).
    rest_pos = rig.local_pos * s
    rest_pos[0] = (rig.root_parent @ np.append(rig.local_pos[0], 1.0))[:3]
    rest_rot = _wxyz(rig.local_quat)
    rest_rot = rest_rot / np.linalg.norm(rest_rot, axis=-1, keepdims=True)
    rest_rot[0] = (parent_rot * Quaternions(rest_rot[0:1])).qs[0]

    tpos_data = {
        'names': np.asarray(rig.names),
        'parents': rig.parents,
        'rest_local_pos': rest_pos,
        'rest_local_rot': rest_rot,
        'fps': np.asarray(30),
    }
    tpos_anim, parents_bfs, names_bfs, _, bfs_order = _load_tpos_npz(tpos_data)

    diameter = float(get_skeleton_diameter(tpos_anim))
    if not np.isfinite(diameter) or diameter <= MIN_SKELETON_DIAMETER:
        raise RigError('degenerate skeleton (zero-length bones)')

    face_joints = None
    if face_pair:
        face_joints = {'r_hip': {'raw': face_pair[0]}, 'l_hip': {'raw': face_pair[1]},
                       'body_axis': body_axis}
    face_idxs, body_axis = resolve_face_joint_idxs(face_joints, names_bfs)
    facing = get_root_facing_quat(positions_global(tpos_anim), face_idxs,
                                  body_axis=body_axis)[0:1]

    canon, xz, ground, scale = canonicalize_anim(
        tpos_anim, face_joint_idxs=face_idxs, body_axis=body_axis,
        target_scale_metric=TARGET_DIAMETER)

    clean_bfs = [clean_names[i] for i in bfs_order]
    cond = build_topology_cond(
        'rig', parents_bfs, np.array(canon.offsets).copy(), names_bfs, clean_bfs,
        positions_global(canon)[0], canon.rotations.qs[0], rotations_global(canon).qs[0],
        max_freqs=max_freqs, face_joint_idxs=face_idxs, body_axis=body_axis,
        scale_factor=scale, ground_height=None,
    )

    return RigContext(
        cond=cond, bfs_order=list(bfs_order), parents_bfs=np.asarray(parents_bfs),
        facing=facing, xz=np.asarray(xz, dtype=np.float64), ground=float(ground),
        scale=float(scale), rest_world_rot=rotations_global(tpos_anim)[0],
        parent_rot=parent_rot, root_parent=rig.root_parent,
    )


def features_to_rig(feats: np.ndarray, ctx: RigContext):
    """Denormalized model features (T, J, 12) in BFS order -> caller's bones."""
    T, J, _ = feats.shape
    parents = ctx.parents_bfs

    rebased = hml_rotations_to_bvh_quaternions(feats[:, :, 3:9], parents)
    _, root_pos_canon = recover_unimate_root_quat_and_pos(feats[:, 0])

    # Re-ground: the model places each clip against its own lowest point, and
    # small errors there sink a walking creature into the floor or float it
    # (measured on the stock rigs: -0.3 to +0.15 of body height). Shift the
    # clip so its lowest joint over all frames touches the rest pose's floor,
    # which is y=0 in the canonical frame. A takeoff still starts on the ground.
    canon_offsets = offsets_from_positions(np.asarray(ctx.cond['tpos_first_frame']), parents)
    lowest = recover_unimate_joint_pos_from_rot(feats, parents, canon_offsets)[..., 1].min()
    root_pos_canon = root_pos_canon.copy()
    root_pos_canon[:, 1] -= lowest

    # World-space deltas from rest, canonical frame. Leaves carry no rotation
    # channel (HML stores each joint's rotation in its child's slot), so their
    # rebased rotation is identity and they follow their parent rigidly.
    delta = Quaternions(rebased.qs.copy())
    for j in range(1, J):
        delta[:, j] = delta[:, parents[j]] * rebased[:, j]

    # Canonical frame -> caller world frame.
    f = ctx.facing
    delta = -f[None] * delta * f[None]

    world = delta * ctx.rest_world_rot[None]
    local = Quaternions(world.qs.copy())
    for j in range(1, J):
        local[:, j] = -world[:, parents[j]] * world[:, j]
    local[:, 0] = -ctx.parent_rot * world[:, 0]

    # Invert canonicalize_anim for the root position: facing -> XZ -> scale -> ground.
    p = root_pos_canon.copy()
    p[:, 1] += ctx.ground
    p = p / ctx.scale + ctx.xz[None]
    p = (-f) * p
    inv_parent = np.linalg.inv(ctx.root_parent)
    root_pos = (inv_parent @ np.concatenate([p, np.ones((T, 1))], axis=1).T).T[:, :3]

    # BFS order -> caller order.
    out_q = np.empty((T, J, 4))
    out_q[:, ctx.bfs_order] = _xyzw(local.normalized().qs)
    return out_q, root_pos

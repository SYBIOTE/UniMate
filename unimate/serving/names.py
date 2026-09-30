"""Joint labels and facing pair for caller rigs.

The model embeds each joint's *cleaned* name with T5 (``model_joint_names``),
so labels are real conditioning, not cosmetics. Named rigs go through the
authors' rule cleaner. Rigs with placeholder names (``bone_12``, ``Bone.023``,
as auto-riggers emit) would all collapse to "Bone"; for those we label joints
from rest-pose geometry instead, assuming the glTF convention of +Z forward
and +Y up.
"""

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from data_process.joint_annotation.names_clean_rule import clean_joint_name, post_process

_RIG_PREFIX_RE = re.compile(r'^(DEF|ORG|MCH|CTRL|MCH-INT)[-_.]', re.IGNORECASE)
_GENERIC_RE = re.compile(r'^(bone|joint|jnt|node)?[\s._-]*\d+$|^(bone|joint)$', re.IGNORECASE)


def clean_names(raw_names: Sequence[str]) -> List[str]:
    return [post_process(clean_joint_name(_RIG_PREFIX_RE.sub('', n), '')) or n
            for n in raw_names]


def looks_generic(raw_names: Sequence[str]) -> bool:
    """True when most names carry no anatomy (auto-rigger placeholders)."""
    generic = sum(bool(_GENERIC_RE.match(n.strip())) for n in raw_names)
    return generic >= 0.5 * len(raw_names)


def _children(parents: np.ndarray) -> List[List[int]]:
    kids = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0:
            kids[p].append(j)
    return kids


_UP = np.array([0.0, 1.0, 0.0])
# Off-midline threshold, as a fraction of the skeleton's bounding diagonal.
# Measured against overall size, not lateral span: a T-posed biped's arm span
# would otherwise put its legs on the midline.
MIDLINE = 0.03


def body_frame(parents: np.ndarray, world_pos: np.ndarray):
    """Infer (right, forward) unit vectors of a Y-up rest pose.

    Rigs arrive facing any horizontal direction, so neither axis is assumed.
    The left-right axis is the horizontal axis of best mirror symmetry; the
    forward sign comes from feet, whose leaf joints (toes, paws) sit ahead of
    their parents on bipeds and quadrupeds alike. Falls back to +Z forward.
    """
    rel = world_pos - world_pos[0]
    scale = np.linalg.norm(np.ptp(world_pos, axis=0)) or 1.0
    best, lateral = np.inf, np.array([1.0, 0.0, 0.0])
    for theta in np.radians(np.arange(0, 180, 2)):
        a = np.array([np.cos(theta), 0.0, np.sin(theta)])
        mirrored = rel - 2 * np.outer(rel @ a, a)
        d = np.linalg.norm(mirrored[:, None] - rel[None], axis=-1).min(axis=1)
        if d.mean() < best:
            best, lateral = d.mean(), a

    forward = np.cross(lateral, _UP)
    kids = _children(parents)
    floor, height = world_pos[:, 1].min(), np.ptp(world_pos[:, 1]) or 1.0
    # Feet only: low leaves off the midline, one vote each. A drooping tail
    # tip is also a low leaf, and it points backward; a long tail bone must
    # not outvote four paws.
    toe_vote = sum(
        np.sign((world_pos[j] - world_pos[parents[j]]) @ forward)
        for j in range(1, len(parents))
        if not kids[j] and world_pos[j, 1] - floor < 0.2 * height
        and abs(rel[j] @ lateral) > MIDLINE * scale
    )
    if toe_vote < 0:
        forward = -forward
    elif toe_vote == 0 and forward[2] < 0:
        forward = -forward
    right = np.cross(forward, _UP)
    return right / np.linalg.norm(right), forward / np.linalg.norm(forward)


def geometric_labels(parents: np.ndarray, world_pos: np.ndarray) -> List[str]:
    """Label joints Left/Right + Leg/Foot/Arm/Hand/Tail/Spine/Neck/Head.

    ``world_pos`` is the (J, 3) Y-up rest pose. Side comes from the mirror
    axis; off-midline chains reaching the floor are legs, the rest arms (or
    wings); midline joints behind the root and not above it are tail, ahead
    of and above it neck/head.
    """
    J = len(parents)
    kids = _children(parents)
    right, forward = body_frame(parents, world_pos)
    rel = world_pos - world_pos[0]
    height = np.ptp(world_pos[:, 1]) or 1.0
    size = np.linalg.norm(np.ptp(world_pos, axis=0)) or 1.0
    floor = world_pos[:, 1].min()

    def subtree(j):
        out, stack = [], [j]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(kids[k])
        return out

    labels = [''] * J
    for j in range(J):
        x = rel[j] @ right
        side = ('Right ' if x > MIDLINE * size else 'Left ' if x < -MIDLINE * size else '')
        is_leaf = not kids[j]
        near_floor = (world_pos[subtree(j), 1].min() - floor) < 0.15 * height
        ahead, up = rel[j] @ forward, rel[j, 1]
        if side and near_floor:
            role = 'Foot' if is_leaf or (world_pos[j, 1] - floor) < 0.1 * height else 'Leg'
        elif side:
            role = 'Hand' if is_leaf else 'Arm'
        elif ahead < -0.05 * height and up <= 0.1 * height:
            role = 'Tail'
        elif ahead > 0.05 * height or up > 0.25 * height:
            role = 'Head' if is_leaf else 'Neck'
        else:
            role = 'Spine' if j else 'Hips'
        labels[j] = side + role
    return labels


def pick_face_pair(names: Sequence[str], labels: Sequence[str], parents: np.ndarray,
                   depth: np.ndarray) -> Optional[Tuple[str, str]]:
    """Shallowest Right/Left joint pair with matching roles -> (right, left) raw names.

    Mirrors stage 3's facing pair (the hips on a biped, the hind legs' roots on
    a quadruped). None means identity facing: the rig is taken to face +Z.
    """
    best = None
    for r, lab_r in enumerate(labels):
        if not lab_r.startswith('Right '):
            continue
        want = 'Left ' + lab_r[len('Right '):]
        for l, lab_l in enumerate(labels):
            if lab_l == want and depth[l] == depth[r] and parents[l] == parents[r]:
                if best is None or depth[r] < best[0]:
                    best = (depth[r], names[r], names[l])
    return (best[1], best[2]) if best else None

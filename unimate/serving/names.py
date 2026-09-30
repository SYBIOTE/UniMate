"""Joint labels and facing pair for caller rigs.

The model embeds each joint's *cleaned* name with T5 (``model_joint_names``),
so labels are real conditioning, not cosmetics. Named rigs go through the
authors' rule cleaner. Rigs with placeholder names (``bone_12``, ``Bone.023``,
as SkinTokens emits for non-humanoids) would all collapse to "Bone"; for those
``analyze_rig`` infers anatomy, the facing pair and the body plan from the
Y-up rest pose. Measured effect of names on the model: body-part prompts
("wags its tail", "lifts its right front leg") target the right joints far
more often with anatomical labels than with "Bone"; whole-body motion depends
on them less.
"""

import re
from dataclasses import dataclass, field
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

# Training vocabulary (UniML3D clean_joint_names): animal limbs are named like
# human ones, segment by segment -- hind legs Thigh > Shin > (Fetlock) > Foot
# > Toe, front legs Shoulder > Upper Arm > Forearm > Hand. Insects use
# whole-limb Front / Middle / Hind Leg.
_LEG = ['Thigh', 'Shin', 'Fetlock', 'Foot', 'Toe', 'Toe End']
_ARM = ['Shoulder', 'Upper Arm', 'Forearm', 'Hand', 'Finger', 'Finger End']


def _segment_names(template: List[str], n: int) -> List[str]:
    """Spread a limb template over an n-bone chain, root to tip."""
    if template is _LEG:
        short = {1: ['Leg'], 2: ['Thigh', 'Shin'], 3: ['Thigh', 'Shin', 'Foot'],
                 4: ['Thigh', 'Shin', 'Foot', 'Toe'], 5: ['Thigh', 'Shin', 'Foot', 'Toe', 'Toe End']}
        pad = 'Toe'
    else:
        short = {1: ['Upper Arm'], 2: ['Upper Arm', 'Forearm'], 3: ['Upper Arm', 'Forearm', 'Hand'],
                 4: ['Shoulder', 'Upper Arm', 'Forearm', 'Hand'],
                 5: ['Shoulder', 'Upper Arm', 'Forearm', 'Hand', 'Finger']}
        pad = 'Finger'
    if n in short:
        return short[n]
    return template[:-1] + [pad] * (n - len(template)) + template[-1:]


@dataclass
class Anatomy:
    """What rest-pose geometry says about a rig. Indices are caller joints."""
    labels: List[str]
    face_pair: Optional[Tuple[int, int]]   # (right, left); (head, tail) when body_axis
    body_axis: bool
    plan: str                              # quadruped | biped | insect | finned | serpentine
    forward: np.ndarray
    # Floor-reaching limbs as joint chains, limb root first, foot (the limb's
    # lowest rest joint) last: what foot-contact cleanup pins.
    leg_chains: List[List[int]] = field(default_factory=list)


def _lateral_axis(rel: np.ndarray) -> np.ndarray:
    """Horizontal axis of best mirror symmetry (bilateral animals)."""
    best, lateral = np.inf, np.array([1.0, 0.0, 0.0])
    for theta in np.radians(np.arange(0, 180, 2)):
        a = np.array([np.cos(theta), 0.0, np.sin(theta)])
        mirrored = rel - 2 * np.outer(rel @ a, a)
        d = np.linalg.norm(mirrored[:, None] - rel[None], axis=-1).min(axis=1).mean()
        if d < best:
            best, lateral = d, a
    return lateral


def analyze_rig(parents: np.ndarray, world_pos: np.ndarray,
                forward_hint: Optional[np.ndarray] = None) -> Anatomy:
    """Infer anatomy from a Y-up rest pose alone (no names).

    1. Left-right axis from mirror symmetry; midline vs sided joints.
    2. Limbs: sided subtrees hanging off the midline, paired across it.
    3. Forward: independent cues vote -- feet point forward; the root's
       branchy side (spine, front limbs, head) is the front and the bare
       chain is the tail; the head end sits higher than the tail end; front
       limbs attach away from the pelvis.
    4. Body plan from floor-reaching limb pairs; labels in the training
       vocabulary; facing pair = the hind leg roots (what training used),
       or the head-tail axis for legless bodies.

    ``forward_hint`` (world, Y-up) settles step 3 when the caller knows which
    way the creature faces -- e.g. from a vision pass over a render of the
    mesh. Only its sign against the symmetry axis is used, so a rough
    direction is enough.
    """
    parents = np.asarray(parents)
    P = np.asarray(world_pos, dtype=np.float64)
    J = len(parents)
    kids = _children(parents)
    rel = P - P[0]
    size = np.linalg.norm(np.ptp(P, axis=0)) or 1.0
    floor, height = P[:, 1].min(), np.ptp(P[:, 1]) or 1.0

    sub = [[] for _ in range(J)]
    for j in sorted(range(J), key=lambda k: -_depth(parents, k)):
        sub[j] = [j] + [d for c in kids[j] for d in sub[c]]

    lateral = _lateral_axis(rel)
    x = rel @ lateral
    thr = MIDLINE * size

    # -- limbs: sibling subtrees that are mirror images of each other across
    # the midline. Proportion-free: a long quadruped's legs sit close under
    # it relative to its length, which defeats any fixed offset threshold.
    cen_all = [rel[sub[j]].mean(0) for j in range(J)]
    mirror = lambda v: v - 2 * (v @ lateral) * lateral
    mid = np.zeros(J, bool)
    limb_root = []
    stack = [0]
    while stack:
        j = stack.pop()
        mid[j] = True
        paired = set()
        for i, c1 in enumerate(kids[j]):
            for c2 in kids[j][i + 1:]:
                if c1 in paired or c2 in paired:
                    continue
                # Centroids mirror when the pose is symmetric. When it is not
                # (one leg raised), identical sub-hierarchies whose roots
                # mirror still mark a pair.
                sep = np.linalg.norm(cen_all[c1] - cen_all[c2])
                err = np.linalg.norm(mirror(cen_all[c1]) - cen_all[c2])
                by_centroid = sep > 0.02 * size and err < 0.25 * sep and x[sub[c1]].mean() * x[sub[c2]].mean() < 0
                rsep = np.linalg.norm(rel[c1] - rel[c2])
                rerr = np.linalg.norm(mirror(rel[c1]) - rel[c2])
                by_root = (len(sub[c1]) == len(sub[c2]) and len(sub[c1]) > 1 and rsep > 0.01 * size
                           and rerr < 0.3 * rsep and x[c1] * x[c2] < 0)
                if by_centroid or by_root:
                    paired.update((c1, c2))
        for c in kids[j]:
            if c in paired or abs(x[sub[c]].mean()) > 2 * thr:
                limb_root.append(c)
            else:
                stack.append(c)
    limb_side = {r: np.sign(x[sub[r]].mean()) for r in limb_root}
    reach_floor = {r: P[sub[r], 1].min() - floor < 0.15 * height for r in limb_root}

    # Pair limbs across the midline by mirrored centroid.
    cen = {r: rel[sub[r]].mean(0) for r in limb_root}
    mirror = lambda v: v - 2 * (v @ lateral) * lateral
    pos_side = [r for r in limb_root if limb_side[r] > 0]
    neg_side = [r for r in limb_root if limb_side[r] < 0]
    pairs = []
    for r in sorted(pos_side, key=lambda k: len(sub[k]), reverse=True):
        if not neg_side:
            break
        l = min(neg_side, key=lambda k: np.linalg.norm(mirror(cen[r]) - cen[k]))
        neg_side.remove(l)
        pairs.append((r, l))

    # -- hub: first fork below the root (the pelvis on SkinTokens rigs).
    hub = 0
    while len(kids[hub]) == 1:
        hub = kids[hub][0]

    # -- forward sign: votes along f0.
    f0 = np.cross(lateral, _UP)
    votes = 0.0
    for r in limb_root:                                   # feet point forward
        if reach_floor[r]:
            toes = [k for k in sub[r] if not kids[k] and k != r and P[k, 1] - floor < 0.2 * height]
            s = sum(np.sign((P[k] - P[parents[k]]) @ f0) for k in toes)
            votes += np.sign(s)
    trunk = [c for c in kids[hub] if c not in limb_side]  # branchy side is the front
    if len(trunk) >= 2:
        big = max(trunk, key=lambda c: len(sub[c]) + 4 * sum(r in sub[c] for r in limb_root))
        small = min(trunk, key=lambda c: len(sub[c]))
        if big != small:
            votes += 2 * np.sign((rel[sub[big]].mean(0) - rel[hub]) @ f0)
    midj = np.flatnonzero(mid)
    ahead0 = rel[midj] @ f0
    a, b = midj[ahead0.argmax()], midj[ahead0.argmin()]
    if abs(P[a, 1] - P[b, 1]) > 0.05 * height:            # head end sits higher
        votes += np.sign(P[a, 1] - P[b, 1])
    att = sum((rel[parents[r]] - rel[hub]) @ f0 for r in limb_root)
    if abs(att) > 0.05 * size:                            # front limbs attach away from the pelvis
        votes += np.sign(att)
    if forward_hint is not None and abs(np.asarray(forward_hint, float) @ f0) > 1e-6:
        votes = float(np.sign(np.asarray(forward_hint, float) @ f0))
    forward = f0 if votes > 0 or (votes == 0 and f0[2] >= 0) else -f0
    right = np.cross(forward, _UP)
    ahead = rel @ forward

    # Order pairs front to back; name sides with the final frame.
    pairs = [(r, l) if cen[r] @ right > 0 else (l, r) for r, l in pairs]
    pairs.sort(key=lambda pr: -(rel[parents[pr[0]]] @ forward))
    floor_pairs = [pr for pr in pairs if reach_floor[pr[0]] or reach_floor[pr[1]]]
    raised_pairs = [pr for pr in pairs if pr not in floor_pairs]
    n_legs = len(floor_pairs)
    plan = ('serpentine' if not limb_root else 'finned' if n_legs == 0 else
            'biped' if n_legs == 1 else 'quadruped' if n_legs == 2 else 'insect')

    labels = ['Bone'] * J
    side_of = lambda r: 'Right ' if cen[r] @ right > 0 else 'Left '

    def label_limb(r, template=None, whole=None):
        # Main path: limb root to its farthest tip; side branches get the tip name.
        tip = max(sub[r], key=lambda k: np.linalg.norm(P[k] - P[r]) + 1e-6 * k)
        path, k = [], tip
        while k != parents[r]:
            path.append(k)
            k = parents[k]
        path.reverse()
        names = [whole] * len(path) if whole else _segment_names(template, len(path))
        extra = whole or ('Toe' if template is _LEG else 'Finger')
        for k in sub[r]:
            labels[k] = side_of(r) + extra
        for k, n in zip(path, names):
            labels[k] = side_of(r) + n

    insect_names = ['Front Leg'] + ['Middle Leg'] * max(n_legs - 2, 0) + ['Hind Leg']
    for i, (rr, ll) in enumerate(floor_pairs):
        for limb in (rr, ll):
            if plan == 'insect':
                label_limb(limb, whole=insect_names[i])
            elif plan == 'quadruped' and i == 0:
                label_limb(limb, _ARM)
            else:
                label_limb(limb, _LEG)
    for rr, ll in raised_pairs:
        for limb in (rr, ll):
            if plan == 'finned':
                label_limb(limb, whole='Fin')
            elif plan == 'quadruped':
                label_limb(limb, whole='Wing')
            else:
                label_limb(limb, _ARM)
    for r in limb_root:
        if all(r not in pr for pr in pairs):
            label_limb(r, _LEG if reach_floor[r] else _ARM)

    # -- midline: root wrapper, hips, spine > neck > head, tail.
    k = 0
    while k != hub:
        labels[k] = 'Root'
        k = kids[k][0]
    labels[hub] = 'Hips'
    mid_rest = [j for j in midj if j != hub and labels[j] == 'Bone']
    head_tip = max(mid_rest, key=lambda j: ahead[j] + 0.5 * rel[j, 1], default=None)
    head_path = []
    if head_tip is not None and ahead[head_tip] > 0:
        k = head_tip
        while k != hub and k >= 0:
            head_path.append(k)
            k = parents[k]
        head_path.reverse()
        attach = [i for i, j in enumerate(head_path) if any(parents[r] == j for r in limb_root)]
        cut = attach[-1] + 1 if attach else len(head_path) // 2
        for i, j in enumerate(head_path):
            labels[j] = 'Spine' if i < cut else 'Neck'
        labels[head_path[-1]] = 'Head'
        if len(head_path) - cut >= 3 and not kids[head_path[-1]]:
            labels[head_path[-2]], labels[head_path[-1]] = 'Head', 'Head End'
    in_head = set(head_path[-2:]) if len(head_path) >= 2 else set(head_path)
    for j in mid_rest:
        if j in head_path:
            continue
        p = parents[j]
        near_head = p in in_head or labels[p] in ('Head', 'Jaw', 'Ear', 'Antenna')
        if near_head:
            labels[j] = ('Jaw' if P[j, 1] < P[p, 1] else 'Head')
        elif ahead[j] < -0.02 * size:
            labels[j] = 'Tail'
        else:
            labels[j] = 'Spine'
    # Sided bits of the head (ears, antennae) are limbs by the rule above; rename.
    for r in limb_root:
        # Front legs of Bip01-style rigs hang off the neck; a limb that
        # reaches the floor is a leg, never an ear.
        if reach_floor[r]:
            continue
        if parents[r] in in_head or labels[parents[r]] in ('Head', 'Head End'):
            for k in sub[r]:
                labels[k] = side_of(r) + ('Antenna' if plan == 'insect' else 'Ear')
    tails = [j for j in range(J) if labels[j] == 'Tail' and not kids[j]]
    for j in tails:
        if labels[parents[j]] == 'Tail':
            labels[j] = 'Tail End'

    # -- facing pair: hind leg roots (the training default), else a fin/wing
    # pair, else the head-tail body axis.
    body_axis = False
    if floor_pairs:
        face = floor_pairs[-1]
    elif raised_pairs:
        face = raised_pairs[0]
    elif head_tip is not None:
        tail_tip = min(midj, key=lambda j: ahead[j])
        face, body_axis = (int(head_tip), int(tail_tip)), True
    else:
        face = None
    leg_chains = []
    for r in limb_root:
        if not reach_floor[r]:
            continue
        foot = min(sub[r], key=lambda k: (round(P[k, 1], 6), -_depth(parents, k)))
        chain, k = [], foot
        while k != parents[r]:
            chain.append(int(k))
            k = parents[k]
        if len(chain) >= 2:
            leg_chains.append(chain[::-1])
    return Anatomy(labels, face, body_axis, plan, forward, leg_chains)


def _depth(parents: np.ndarray, j: int) -> int:
    d = 0
    while parents[j] >= 0:
        j, d = parents[j], d + 1
    return d


def geometric_labels(parents: np.ndarray, world_pos: np.ndarray) -> List[str]:
    return analyze_rig(parents, world_pos).labels


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

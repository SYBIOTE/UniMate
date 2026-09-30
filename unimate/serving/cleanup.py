"""Post-processing on generated motion, in the canonical frame (Y-up, floor y=0,
T-pose-rebased rotations, so every joint's rest rotation is identity).

- ``foot_lock``: detect foot contacts and pin each contact to one spot on the
  floor with CCD IK over the leg, removing foot sliding and the sideways splay
  that comes with it.
- ``rotate_loop_to_free_frame`` / ``close_seam``: keep a loop's seam intact
  under foot lock.

Quaternions here are plain (..., 4) w, x, y, z arrays: the Holden Quaternions
class broadcasts too strictly for per-frame, per-joint batch updates.
"""

from typing import List, Sequence

import numpy as np

# Contact thresholds, in canonical units (the skeleton is scaled to a leaf-to-leaf
# diameter of 2): a foot within 6% of that of the floor, moving under ~0.6 units/s.
CONTACT_HEIGHT = 0.12
CONTACT_SPEED = 0.02          # per frame at 30 fps
MIN_CONTACT_FRAMES = 3
BLEND_FRAMES = 3
IK_ITERATIONS = 10


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qinv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[..., 1:], q[..., :1]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def qbetween(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation taking direction a onto direction b."""
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    q = np.concatenate([1.0 + np.sum(a * b, axis=-1, keepdims=True), np.cross(a, b)], axis=-1)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)


def qscale(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Slerp from identity toward q by w in [0, 1] (per leading index)."""
    q = np.where(q[..., :1] < 0, -q, q)
    angle = 2.0 * np.arccos(np.clip(q[..., 0], -1.0, 1.0))
    axis = q[..., 1:] / np.maximum(np.sin(angle / 2.0)[..., None], 1e-9)
    half = angle * w / 2.0
    return np.concatenate([np.cos(half)[..., None], axis * np.sin(half)[..., None]], axis=-1)


def forward_kinematics(local: np.ndarray, root_pos: np.ndarray, parents: Sequence[int], offsets: np.ndarray):
    """(T, J, 4) local rotations -> (T, J, 4) global rotations, (T, J, 3) positions."""
    T, J = local.shape[:2]
    glob = np.empty_like(local)
    pos = np.empty((T, J, 3))
    glob[:, 0], pos[:, 0] = local[:, 0], root_pos
    for j in range(1, J):
        p = parents[j]
        glob[:, j] = qmul(glob[:, p], local[:, j])
        pos[:, j] = pos[:, p] + qrot(glob[:, p], np.broadcast_to(offsets[j], (T, 3)))
    return glob, pos


def _contact_segments(foot: np.ndarray) -> List[range]:
    """Runs of frames where a foot (T, 3) is low and nearly still."""
    speed = np.zeros(len(foot))
    speed[1:] = np.linalg.norm(np.diff(foot[:, [0, 2]], axis=0), axis=-1)
    speed[0] = speed[1] if len(foot) > 1 else 0.0
    on = (foot[:, 1] < CONTACT_HEIGHT) & (speed < CONTACT_SPEED)
    segments, start = [], None
    for t, c in enumerate(np.append(on, False)):
        if c and start is None:
            start = t
        elif not c and start is not None:
            if t - start >= MIN_CONTACT_FRAMES:
                segments.append(range(start, t))
            start = None
    return segments


def _contact_weights(pos: np.ndarray, leg_chains: Sequence[Sequence[int]]) -> np.ndarray:
    """(T,) max over feet of the pin weight foot_lock would apply, blend included."""
    T = len(pos)
    weight = np.zeros(T)
    for chain in leg_chains:
        for seg in _contact_segments(pos[:, chain[-1]]):
            lo, hi = max(seg.start - BLEND_FRAMES, 0), min(seg.stop + BLEND_FRAMES, T)
            weight[lo:hi] = 1.0
    return weight


def rotate_loop_to_free_frame(local: np.ndarray, root_pos: np.ndarray, parents: Sequence[int],
                              offsets: np.ndarray, leg_chains: Sequence[Sequence[int]]):
    """Restart a closed cycle (frame L repeats frame 0) at a frame where no foot is pinned.

    foot_lock pins each contact on its own, so a contact straddling the seam
    would be solved differently at frame 0 and frame L and break the loop.
    Starting the cycle where no foot is planted (nor blending in) keeps both
    ends untouched by the IK, hence identical. Returns (local, root_pos, shift);
    shift 0 means no free frame existed, or the start already was one.
    """
    L = len(local) - 1
    _, pos = forward_kinematics(local, root_pos, parents, offsets)
    busy = _contact_weights(pos[:L], leg_chains)
    if not busy.any() or not busy[0]:
        return local, root_pos, 0
    free = np.flatnonzero(busy == 0)
    if not len(free):
        return local, root_pos, 0
    # Middle of the longest free run: furthest from any blend.
    runs = np.split(free, np.flatnonzero(np.diff(free) != 1) + 1)
    shift = int(max(runs, key=len)[len(max(runs, key=len)) // 2])
    drift = root_pos[L] - root_pos[0]
    order = np.r_[shift:L, 0:shift + 1]
    new_root = root_pos[order].copy()
    new_root[L - shift:] += drift          # wrapped frames continue past the old end
    return local[order].copy(), new_root, shift


def foot_lock(local: np.ndarray, root_pos: np.ndarray, parents: Sequence[int],
              offsets: np.ndarray, leg_chains: Sequence[Sequence[int]]) -> np.ndarray:
    """Pin feet during contacts; returns new (T, J, 4) local rotations.

    Each contact is pinned where the foot was on average during it, at no less
    than floor height, easing in and out over BLEND_FRAMES so the leg does not
    snap. The IK only rotates the leg's own joints (limb root to the foot's
    parent); body and root motion are untouched.
    """
    local = local.copy()
    T = local.shape[0]
    _, pos = forward_kinematics(local, root_pos, parents, offsets)

    for chain in leg_chains:
        foot = chain[-1]
        segments = _contact_segments(pos[:, foot])
        if not segments:
            continue
        target = pos[:, foot].copy()
        weight = np.zeros(T)
        for seg in segments:
            idx = np.array(seg)
            spot = pos[idx, foot].mean(axis=0)
            spot[1] = max(spot[1], pos[idx, foot, 1].min(), 0.0)
            target[idx] = spot
            weight[idx] = 1.0
            # Ease in/out, holding the same spot, so the foot settles into place.
            for k in range(1, BLEND_FRAMES + 1):
                w = 1.0 - k / (BLEND_FRAMES + 1)
                for t in (idx[0] - k, idx[-1] + k):
                    if 0 <= t < T and weight[t] < w:
                        target[t], weight[t] = spot, w
        frames = np.flatnonzero(weight > 0)
        joints = list(chain[:-1])
        for _ in range(IK_ITERATIONS):
            for j in reversed(joints):
                glob, pos = forward_kinematics(local, root_pos, parents, offsets)
                a = pos[frames, foot] - pos[frames, j]
                b = target[frames] - pos[frames, j]
                q = qscale(qbetween(a, b), weight[frames])
                new_glob = qmul(q, glob[frames, j])
                local[frames, j] = qmul(qinv(glob[frames, parents[j]]), new_glob)
    return local


def qslerp(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Slerp from a to b by w (broadcast over leading dims)."""
    b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -b, b)
    return qmul(a, qscale(qmul(qinv(a), b), w))


def close_seam(local: np.ndarray, frames: int = BLEND_FRAMES + 2) -> np.ndarray:
    """Force frame L to equal frame 0, easing the last few frames into it.

    Fallback for loops where the seam could not be moved clear of every foot
    contact, so the IK left the two ends different.
    """
    local = local.copy()
    L = len(local) - 1
    for i in range(frames, 0, -1):
        t = L - i + 1
        w = 1.0 - (i - 1) / frames
        local[t] = qslerp(local[t], local[0], np.full(local.shape[1], w))
    return local

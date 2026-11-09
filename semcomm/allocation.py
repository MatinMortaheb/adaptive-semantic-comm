"""Rate allocation algorithms.

* binary_select        : compressor of the binary scheme (alpha-controlled attention + random fill)
* alg2_legacy          : Algorithm 2 as published in the conference version / submitted manuscript
                         (kept ONLY to measure how often and by how much it exceeds the budget)
* alg2_feasible        : corrected, budget-feasible Algorithm 2 (Proposition 1)
* uniform_levels       : uniform-rate patch coding baseline (no importance)
* iaq_allocate         : importance-aware scalar quantization baseline of Park et al. (IoT-J 2025)

All budgets are in BYTES of payload unless stated otherwise.
"""
import heapq
import math

import numpy as np


# ----------------------------------------------------------------- binary
def binary_select(scores, n_budget, alpha, rng):
    """scores: (P,) importance; n_budget: number of patches that fit the budget.
    Top floor(alpha*n) patches by attention, the rest drawn uniformly from the remaining ones.
    alpha=1: pure attention; alpha=0: purely random selection ( baseline)."""
    P = len(scores)
    n_budget = int(min(max(n_budget, 0), P))
    n_att = int(math.floor(alpha * n_budget + 1e-9))
    order = np.argsort(-scores, kind="stable")
    sel = np.zeros(P, dtype=bool)
    sel[order[:n_att]] = True
    rest = order[n_att:]
    k = n_budget - n_att
    if k > 0:
        sel[rng.choice(rest, size=k, replace=False)] = True
    return sel


# ------------------------------------------------------------- helpers
def _lq_level(a, R):
    """index of the largest level whose size <= a (LQ of Algorithm 3)."""
    R = np.asarray(R)
    return np.clip(np.searchsorted(R, a, side="right") - 1, 0, len(R) - 1)


# ------------------------------------------------- Algorithm 2 (legacy)
def alg2_legacy(scores, r, R=(0, 12, 24, 48, 192), eps=1e-6, max_iter=100000):
    """Faithful re-implementation of the submitted Algorithm 2 (with 192 instead of the 196 typo).
    Returns level indices. May exceed r (that is the point of keeping it)."""
    R = np.asarray(R, dtype=float)
    P = len(scores)
    A = scores.astype(float).copy()
    A = A * (r / max(A.sum(), 1e-12))
    order = np.argsort(-A, kind="stable")
    if r <= R[1] * P:
        lv = np.zeros(P, dtype=int)
        lv[order[: int(r // R[1])]] = 1
        return lv
    lq = _lq_level(A, R)
    LQ = R[lq].copy()
    uq = np.minimum(lq + 1, len(R) - 1)
    UQ = R[uq].copy()
    A[LQ == R[-1]] = -np.inf
    LQ[LQ == 0] = R[1]
    it = 0
    while LQ.sum() < r and it < max_iter:
        it += 1
        diff = UQ - A
        j = int(np.argmin(diff))
        if not np.isfinite(diff[j]):
            break
        LQ[j] = UQ[j]
        A[j] = LQ[j] + eps
        if UQ[j] == R[-1]:
            A[j] = -np.inf
        u = np.minimum(np.searchsorted(R, A, side="right"), len(R) - 1)
        UQ = R[u]
    return np.searchsorted(R, LQ)


# ---------------------------------------------- Algorithm 2 (feasible)
def alg2_feasible(scores, r, R=(0, 12, 24, 48, 192), side_bytes=0.0, return_info=False):
    """Budget-feasible Algorithm 2 (revised manuscript). Guarantees
    sum_i R[l_i] + side_bytes <= r  (Proposition 1). O(P L log P)."""
    R = np.asarray(R, dtype=float)
    L = len(R)
    P = len(scores)
    rp = r - side_bytes
    lv = np.zeros(P, dtype=int)
    if rp <= 0:
        return (lv, {"U": 0.0, "budget": rp}) if return_info else lv
    order = np.argsort(-scores, kind="stable")
    if rp < R[1] * P:  # binary regime
        lv[order[: int(rp // R[1])]] = 1
        U = R[lv].sum()
        return (lv, {"U": U, "budget": rp}) if return_info else lv
    a = scores.astype(float) * (rp / max(scores.sum(), 1e-12))
    lv = np.maximum(1, _lq_level(a, R))
    U = R[lv].sum()
    # repair: downgrade least-important upgraded patches until feasible
    if U > rp:
        for j in np.argsort(a, kind="stable"):
            while lv[j] > 1 and U > rp:
                U -= R[lv[j]] - R[lv[j] - 1]
                lv[j] -= 1
            if U <= rp:
                break
    # feasible upgrades, smallest gap to the next level first
    heap = []
    for i in range(P):
        if lv[i] < L - 1:
            heapq.heappush(heap, (R[lv[i] + 1] - max(a[i], R[lv[i]]), -a[i], i, lv[i]))
    while heap:
        gap, _, i, l_at = heapq.heappop(heap)
        if lv[i] != l_at:
            continue
        cost = R[lv[i] + 1] - R[lv[i]]
        if cost <= rp - U + 1e-9:
            U += cost
            lv[i] += 1
            if lv[i] < L - 1:
                heapq.heappush(heap, (R[lv[i] + 1] - max(a[i], R[lv[i]]), -a[i], i, lv[i]))
        # else: remaining budget only shrinks -> this upgrade can never become feasible; drop it
    assert U <= rp + 1e-6
    return (lv, {"U": U, "budget": rp}) if return_info else lv


# ------------------------------------------------- uniform-rate baseline
def uniform_levels(P, r, R=(0, 12, 24, 48, 192), rng=None):
    """No importance: all patches at the largest level that fits; the leftover budget upgrades a
    uniformly random subset of patches to the next level (time-sharing between adjacent levels)."""
    R = np.asarray(R, dtype=float)
    rng = rng or np.random.default_rng(0)
    if r <= 0:
        return np.zeros(P, dtype=int)
    base = int(np.searchsorted(R * P, r, side="right") - 1)
    base = max(0, min(base, len(R) - 1))
    lv = np.full(P, base, dtype=int)
    if base == 0:  # cannot afford level 1 for all: random subset at level 1
        n = int(r // R[1])
        lv[rng.choice(P, size=min(n, P), replace=False)] = 1
        return lv
    if base < len(R) - 1:
        n_up = int((r - R[base] * P) // (R[base + 1] - R[base]))
        lv[rng.choice(P, size=min(n_up, P), replace=False)] = base + 1
    return lv


# --------------------------------------------------------- IAQ baseline
def quantize_patch_pixels(x, M):
    """Uniform mid-rise quantization of values in [0,1] with M bits (M=0 -> constant 0.5)."""
    if M == 0:
        return np.full_like(x, 0.5)
    Lv = 2 ** M
    return (np.floor(np.clip(x, 0, 1 - 1e-9) * Lv) + 0.5) / Lv


def iaq_allocate(patches, scores, budget_bits, max_bits=8):
    """Park et al. (IoT-J 2025) importance-aware quantization, incremental allocation.
    patches: (P, n) pixel values in [0,1]; scores: (P,) attention; budget_bits for the payload.
    Minimizes sum_i w_i * MSE_i(M_i) s.t. sum_i n*M_i <= budget, w_i = normalized attention.
    Greedy incremental allocation (optimal for convex decreasing D_i(M))."""
    P, n = patches.shape
    w = scores / max(scores.sum(), 1e-12)
    D = np.stack([((quantize_patch_pixels(patches, m) - patches) ** 2).mean(1) for m in range(max_bits + 1)], 1)
    M = np.zeros(P, dtype=int)
    used = 0
    heap = [(-(w[i] * (D[i, 0] - D[i, 1])), i) for i in range(P)]
    heapq.heapify(heap)
    while heap and used + n <= budget_bits:
        g, i = heapq.heappop(heap)
        M[i] += 1
        used += n
        if M[i] < max_bits:
            heapq.heappush(heap, (-(w[i] * (D[i, M[i]] - D[i, M[i] + 1])), i))
    return M

"""Property tests for the allocation algorithms (no GPU / dataset needed).

    python tests/test_allocation.py          # or: pytest tests/

Checks Proposition 1 (feasibility, coverage, maximality) for the corrected Algorithm 2 on many
synthetic attention maps, and measures how often the legacy rule exceeds the budget."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semcomm import allocation as al  # noqa: E402
from semcomm import rate as rt  # noqa: E402

R = (0, 12, 24, 48, 192)


def synthetic_attention(rng, gh=40, gw=60, kind="blobs"):
    """Object-like attention maps: a few Gaussian blobs + heavy-tailed background (positive, sums to 1)."""
    if kind == "dirichlet":
        a = rng.dirichlet(np.full(gh * gw, 0.3))
        return a
    yy, xx = np.mgrid[0:gh, 0:gw]
    a = rng.gamma(0.5, 0.02, size=(gh, gw))
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.uniform(0, gh), rng.uniform(0, gw)
        s = rng.uniform(2, 10)
        a += rng.uniform(0.5, 2) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    a = a.ravel()
    return a / a.sum()


def check_feasible(n=300, seed=0, P_grid=(40, 60)):
    rng = np.random.default_rng(seed)
    P = P_grid[0] * P_grid[1]
    side = rt.mr_side_bits_fixed(P, len(R), 16) / 8
    fails = 0
    for t in range(n):
        a = synthetic_attention(rng, *P_grid, kind="blobs" if t % 2 else "dirichlet")
        r = rng.uniform(0.01, 1.05) * P * R[-1]
        lv, info = al.alg2_feasible(a, r, R, side_bytes=side, return_info=True)
        U = np.asarray(R)[lv].sum()
        ok = U + side <= r + 1e-6
        if r - side >= R[1] * P:
            ok &= (lv >= 1).all()
            rem = r - side - U
            up = [R[l + 1] - R[l] for l in lv if l < len(R) - 1]
            ok &= (not up) or rem < min(up)
        fails += not ok
    return fails


def legacy_overshoot(n=300, seed=1, fracs=(0.0625, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75)):
    rng = np.random.default_rng(seed)
    P = 2400
    rows = []
    for f in fracs:
        over = []
        for t in range(n):
            a = synthetic_attention(rng, kind="blobs" if t % 2 else "dirichlet")
            r = f * P * R[-1]
            lv = al.alg2_legacy(a, r, R)
            over.append(np.asarray(R)[lv].sum() - r)
        over = np.array(over)
        rows.append((f, (over > 1e-6).mean(), np.clip(over, 0, None).mean(), over.max()))
    return rows


def test_feasible():
    assert check_feasible(120) == 0


def test_binary_select():
    rng = np.random.default_rng(0)
    s = rng.random(100)
    for a in (0, 0.5, 1):
        sel = al.binary_select(s, 30, a, rng)
        assert sel.sum() == 30
        top = np.argsort(-s)[: int(a * 30)]
        assert sel[top].all()


if __name__ == "__main__":
    print("feasibility violations (corrected Alg. 2):", check_feasible())
    print("legacy Alg. 2 on synthetic attention maps (P=2400):")
    print(" budget/R_full | frac. over budget | mean overshoot (B) | max overshoot (B)")
    for f, fr, m, mx in legacy_overshoot():
        print(f" {f:13g} | {fr:17.3f} | {m:18.1f} | {mx:17.0f}")

"""Bit-level rate accounting (revised Methods: Rate accounting, Eqs. (16)-(18))."""
import math

import numpy as np


def binary_mask_bits(P, n):
    """min(bitmap, enumerative code of the selected set + its cardinality)."""
    enum = math.ceil((math.lgamma(P + 1) - math.lgamma(n + 1) - math.lgamma(P - n + 1)) / math.log(2)) \
        + math.ceil(math.log2(P))
    return min(P, enum)


def binary_rate_bits(P, n, D=384, token_bits=8):
    return n * D * token_bits + binary_mask_bits(P, n)


def mr_side_bits_fixed(P, L, header_bits=16):
    return P * math.ceil(math.log2(L)) + header_bits


def mr_side_bits_entropy(levels, L, header_bits=16):
    """Empirical-entropy estimate of an arithmetic-coded resolution map (+ histogram + header)."""
    levels = np.asarray(levels).ravel()
    P = levels.size
    cnt = np.bincount(levels, minlength=L).astype(float)
    p = cnt[cnt > 0] / P
    H = -(p * np.log2(p)).sum()
    hist_bits = L * math.ceil(math.log2(P + 1))
    return math.ceil(P * H) + hist_bits + header_bits


def mr_rate_bits(levels, R, header_bits=16, entropy_coded=False):
    levels = np.asarray(levels)
    L = len(R)
    payload = 8 * np.asarray(R)[levels].sum()
    side = mr_side_bits_entropy(levels, L, header_bits) if entropy_coded else mr_side_bits_fixed(levels.size, L, header_bits)
    return payload + side, payload, side


def bpp(bits, native_hw=(64, 64)):
    return bits / (native_hw[0] * native_hw[1])


def full_budget_bytes(P, raw=192):
    return P * raw


# ------------------------------------------------------- channel uses (WITT comparison)
def bits_for_channel_uses(n_cu, eta):
    return n_cu * eta


def bandwidth_ratio(n_cu, native_hw=(64, 64)):
    """rho = channel uses / number of real source samples (3*h0*w0), WITT's CBR convention."""
    return n_cu / (3 * native_hw[0] * native_hw[1])

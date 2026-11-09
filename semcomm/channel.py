"""Channel / rate-trace model for the robustness study (revised Methods: Channel and rate-trace model)."""
import numpy as np

# 3GPP TS 38.214 Table 5.2.2.1-2 (4-bit CQI table 1) spectral efficiencies [bit/s/Hz = bit/channel use]
CQI_EFF = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
                    2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])
# Approximate AWGN SNR thresholds (dB) for 10% BLER per CQI, as commonly used in link-to-system
# mapping. A configurable margin (channel.mcs_margin_db) is added for lower-BLER operation.
CQI_SNR_DB = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7])


def capacity_eta(snr_db):
    return np.log2(1 + 10 ** (np.asarray(snr_db) / 10))


def mcs_eta(snr_db, margin_db=1.0):
    """Spectral efficiency of the highest CQI whose threshold (+margin) is below snr_db (0 if none)."""
    snr_db = np.atleast_1d(np.asarray(snr_db, dtype=float))
    idx = np.searchsorted(CQI_SNR_DB + margin_db, snr_db, side="right") - 1
    eta = np.where(idx >= 0, CQI_EFF[np.clip(idx, 0, None)], 0.0)
    return eta if eta.size > 1 else float(eta[0])


def rayleigh_trace(n_blocks, avg_snr_db, rng):
    """Block fading: |h|^2 ~ Exp(1), constant within a block. Returns instantaneous SNR in dB."""
    g = rng.exponential(1.0, size=n_blocks)
    return 10 * np.log10(np.maximum(g * 10 ** (avg_snr_db / 10), 1e-12))


def estimated_snr_db(snr_db, sigma, rng):
    """gamma_hat = gamma * (1 + eps), eps ~ N(0, sigma^2) (relative error on linear SNR)."""
    lin = 10 ** (np.asarray(snr_db) / 10)
    eps = rng.normal(0.0, sigma, size=np.shape(snr_db)) if sigma > 0 else 0.0
    return 10 * np.log10(np.maximum(lin * (1 + eps), 1e-12))


def block_budget_bits(n_cu, snr_db, snr_hat_db, margin_db=1.0, outage_model="truncate"):
    """Returns (planned_bits, delivered_bits) for one block.
    planned  : what the allocator was told (from the estimated SNR)
    delivered: 'truncate'   -> min(planned, n_cu*eta(true SNR)) (segmented code blocks, in-order delivery)
               'drop_block' -> 0 if the chosen MCS is not supported by the true SNR, else planned"""
    planned = n_cu * mcs_eta(snr_hat_db, margin_db)
    supported = n_cu * mcs_eta(snr_db, margin_db)
    if planned <= supported + 1e-9:
        return planned, planned
    return planned, (supported if outage_model == "truncate" else 0.0)


def packet_erasure_mask(n_packets, p_e, rng):
    """True = packet received."""
    return rng.random(n_packets) >= p_e

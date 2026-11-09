import numpy as np
import torch


def psnr(x, y):
    """per-image PSNR (dB) for tensors in [0,1], shape (B,3,H,W)."""
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def masked_mse(x, y, mask_pix):
    """MSE over pixels where mask_pix (B,1,H,W) is 1 (paper's 'masked MSE')."""
    m = mask_pix.expand_as(x)
    num = (((x - y) ** 2) * m).flatten(1).sum(1)
    den = m.flatten(1).sum(1).clamp_min(1)
    return num / den


def ms_ssim(x, y):
    from pytorch_msssim import ms_ssim as _ms
    H, W = x.shape[-2:]
    if min(H, W) > 160:
        return _ms(x, y, data_range=1.0, size_average=False)
    from pytorch_msssim import ssim
    return ssim(x, y, data_range=1.0, size_average=False, win_size=7)


def bootstrap_ci(values, n=10000, alpha=0.05, seed=0):
    """mean and percentile 95% CI of the mean."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    chunk = max(1, int(2e7 // v.size))  # bounded memory for large samples
    means = np.concatenate([v[rng.integers(0, v.size, size=(min(chunk, n - i), v.size))].mean(1)
                            for i in range(0, n, chunk)])
    return v.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)

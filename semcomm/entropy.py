"""Entropy coding of the quantized patch-codec symbols (rates comparable to standard codecs).

Model: a static, non-parametric factorized probability model -- one 256-bin pmf per latent dimension
and per resolution level, estimated on training patches (add-one smoothing). This is the
non-parametric counterpart of the factorized entropy model of learned codecs (Balle et al.).
It is shared by encoder and decoder, so it costs no side information.

Rate: the ideal code length  sum_d -log2 p_d(q_d)  per patch. An arithmetic / range coder driven by
the same model attains this within a few bytes per image, so it is reported as the entropy-coded rate.
The raw 192-byte level is entropy-coded the same way (one pmf per pixel position and colour).
"""
import numpy as np
import torch


class EntropyModel:
    def __init__(self, counts, device="cpu"):
        """counts: dict size(int) -> array (dims, 256) of symbol counts."""
        self.counts = {int(b): np.asarray(c, dtype=np.float64) for b, c in counts.items()}
        self.nlog2p = {}
        for b, c in self.counts.items():
            p = (c + 1.0) / (c + 1.0).sum(1, keepdims=True)
            self.nlog2p[b] = torch.tensor(-np.log2(p), dtype=torch.float32, device=device)

    def to(self, device):
        for b in self.nlog2p:
            self.nlog2p[b] = self.nlog2p[b].to(device)
        return self

    def bits(self, b, sym):
        """sym: (N, dims) long in [0,255] -> (N,) bits."""
        t = self.nlog2p[int(b)]
        return t[torch.arange(t.shape[0], device=sym.device), sym.long()].sum(1)

    def mean_bytes(self, b):
        """expected coded size (bytes per patch) under the training distribution."""
        c = self.counts[int(b)]
        p = (c + 1.0) / (c + 1.0).sum(1, keepdims=True)
        emp = c / np.maximum(c.sum(1, keepdims=True), 1)
        return float((emp * -np.log2(p)).sum() / 8.0)

    def save(self, path):
        np.savez_compressed(path, **{str(b): c for b, c in self.counts.items()})

    @staticmethod
    def load(path, device="cpu"):
        z = np.load(path)
        return EntropyModel({int(k): z[k] for k in z.files}, device)

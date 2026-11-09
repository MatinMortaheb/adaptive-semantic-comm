"""S9 - Transmitter complexity and latency.
Parameters, GFLOPs (fvcore) and median latency (batch 1) of each transmitter component.
Run on the GPU and, if available, on the edge device you want to quote (e.g. Jetson).
MAFT+ (MMSC extractor) is measured from your MAFT+ install if --maft_cfg is given; otherwise
only its parameter count from the literature is left for you to fill in."""
from _common import setup

import time

import numpy as np
import pandas as pd
import torch

from semcomm import allocation as alloc
from semcomm.data import patchify, to_input
from semcomm.models import CodecBank, ViTWithHead
from semcomm.utils import out_path


def latency(fn, dev, n=50, warm=10):
    for _ in range(warm):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t))
    return float(np.median(ts))


def gflops(model, inp):
    """GFLOPs (1 multiply-add = 2 FLOPs) with PyTorch's built-in FLOP counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(*inp)
        return fc.get_total_flops() / 1e9
    except Exception as e:  # noqa
        print("[warn] flop counting failed:", e)
        return float("nan")


class _Attn(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit.tokens_and_attention(x)[1]


def main():
    args, cfg, dev = setup(__doc__)
    p = cfg["vit"]["patch"]
    H, W = cfg["data"]["input_hw"]
    cfg["vit"]["pretrained"] = False  # weights irrelevant for cost
    x_nat = torch.rand(1, 3, *cfg["data"]["native_hw"], device=dev)
    x_in = to_input(x_nat, cfg)
    rows = []
    vit = ViTWithHead(cfg).to(dev).eval()
    with torch.no_grad():
        rows.append(dict(component=f"ViT-S/8 importance ({H}x{W})", params_M=sum(q.numel() for q in vit.parameters()) / 1e6,
                         gflops=gflops(_Attn(vit), (x_in,)), latency_ms=latency(lambda: vit.tokens_and_attention(x_in), dev)))
        s = vit.tokens_and_attention(x_in)[1][0, :, 0, 1:].mean(0).double().cpu().numpy()
        R = list(cfg["mr"]["levels_bytes"])
        P = len(s)
        rows.append(dict(component="Resolution selector (Alg. 2, CPU)", params_M=0.0, gflops=float("nan"),
                         latency_ms=latency(lambda: alloc.alg2_feasible(s, 0.25 * P * R[-1], R, 902), torch.device("cpu"), n=20, warm=2)))
        bank = CodecBank(R, p, cfg["mr"]["latent_bits"]).to(dev).eval()
        pt = patchify(x_in, p).flatten(0, 1)
        enc_params = sum(q.numel() for c in bank.codecs.values() for q in c.enc.parameters()) / 1e6
        for b, c in bank.codecs.items():
            rows.append(dict(component=f"Patch encoder {b} B (all {P} patches)", params_M=sum(q.numel() for q in c.enc.parameters()) / 1e6,
                             gflops=gflops(c.enc, (pt,)), latency_ms=latency(lambda c=c: c.enc(pt), dev, n=20)))
        rows.append(dict(component="Patch encoders, all levels (params)", params_M=enc_params, gflops=float("nan"), latency_ms=float("nan")))
    df = pd.DataFrame(rows)
    df["device"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    df.to_csv(out_path(cfg, "results", "s9_complexity.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df)


if __name__ == "__main__":
    main()

"""S2b - Fit the entropy model of the patch codecs (entropy coding).

For every level of the codec bank (12/24/48 B learned codecs and the raw 192 B level), counts the
8-bit symbols that the encoder transmits, per latent dimension, on training patches, and saves
outputs/ckpt/entropy_<domain>.npz. Afterwards every script that uses the codec bank (S3, S4, S6, S7,
S8) reports entropy-coded rates automatically (disable with --set rate.entropy_coding=false).
Run after s2_train_codecs.py (and again whenever the codecs are retrained)."""
from _common import setup

import os

import numpy as np
import torch

from semcomm.data import get_loader, patchify, to_input
from semcomm.entropy import EntropyModel
from semcomm.system import load_codec_bank
from semcomm.utils import limited, out_path, save_json


def extra(ap):
    ap.add_argument("--domain", default="upsampled", choices=["upsampled", "native"])
    ap.add_argument("--images", type=int, default=20000, help="training images used to fit the pmfs")


def main():
    args, cfg, dev = setup(__doc__, extra)
    m, p = cfg["mr"], cfg["vit"]["patch"]
    sizes = sorted({b for b in list(m["levels_bytes"]) + list(m["ablation_sizes"]) if b != 0})
    cfg["rate"] = {"entropy_coding": False}
    bank = load_codec_bank(cfg, dev, sizes=sizes, domain=args.domain)
    sizes = [b for b in sizes if b == bank.raw or os.path.exists(out_path(cfg, "ckpt", f"codec_{args.domain}_{b}.pt"))]
    print("fitting entropy model for levels:", sizes)
    counts = {b: np.zeros((b if b != bank.raw else 3 * p * p, 256), dtype=np.int64) for b in sizes}
    bs = cfg["train"]["batch_size"]
    dl = get_loader(cfg, "train", batch_size=bs)
    seen = 0
    with torch.no_grad():
        for x, _ in limited(dl, None):
            xc = to_input(x.to(dev), cfg) if args.domain == "upsampled" else x.to(dev)
            pt = patchify(xc, p).flatten(0, 1)
            if args.domain == "upsampled":  # 2400 patches/image: subsample for speed (unbiased)
                pt = pt[torch.randperm(pt.shape[0], device=pt.device)[: 8192]]
            for b in sizes:
                sym = bank.symbols(pt, b)  # (N, dims)
                d = sym.shape[1]
                idx = (torch.arange(d, device=dev) * 256).unsqueeze(0) + sym
                counts[b] += torch.bincount(idx.flatten(), minlength=d * 256).view(d, 256).cpu().numpy()
            seen += x.shape[0]
            if seen >= args.images:
                break
    em = EntropyModel(counts)
    path = out_path(cfg, "ckpt", f"entropy_{args.domain}.npz")
    em.save(path)
    summ = {str(b): {"nominal_bytes": float(b), "entropy_bytes_per_patch": em.mean_bytes(b),
                     "ratio": em.mean_bytes(b) / b} for b in sizes}
    save_json(summ, out_path(cfg, "results", f"s2b_entropy_{args.domain}.json"))
    for b, v in summ.items():
        print(f"[{args.domain}] level {b:>3} B -> {v['entropy_bytes_per_patch']:.2f} B/patch entropy-coded ({100 * v['ratio']:.0f}%)")


if __name__ == "__main__":
    main()

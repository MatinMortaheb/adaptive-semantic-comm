"""S2 - Train the per-resolution patch encoder/decoder pairs (Fig. 22(b)) with per-patch MSE,
8-bit quantized latents (straight-through). Trains every size needed by the main level set and
by the level-set ablation, and records parameter counts and training time (overhead).

--domain upsampled : patches of the 320x480 input (paper)
--domain native    : patches of the native 64x64 image (used by MR-native, see README)"""
from _common import setup

import os

import torch
import torch.nn.functional as F

from semcomm.data import get_loader, patchify, to_input
from semcomm.models import PatchCodec
from semcomm.utils import Timer, limited, load_json, out_path, save_ckpt, save_json


def extra(ap):
    ap.add_argument("--domain", default=None, choices=["upsampled", "native"])
    ap.add_argument("--sizes", type=int, nargs="*", default=None)


def main():
    args, cfg, dev = setup(__doc__, extra)
    m, tr, p = cfg["mr"], cfg["train"], cfg["vit"]["patch"]
    domain = args.domain or m["codec_domain"]
    raw = 3 * p * p
    sizes = args.sizes or sorted({b for b in list(m["levels_bytes"]) + list(m["ablation_sizes"]) if b not in (0, raw)})
    res_path = out_path(cfg, "results", f"s2_codecs_{domain}.json")
    results = load_json(res_path) if os.path.exists(res_path) else {}
    train_dl = get_loader(cfg, "train", batch_size=m["batch_size"])
    test_dl = get_loader(cfg, "test", batch_size=m["batch_size"], shuffle=False)
    for b in sizes:
        torch.manual_seed(0)
        codec = PatchCodec(b, p, m["latent_bits"], m["latent_activation"]).to(dev)
        opt = torch.optim.Adam(codec.parameters(), lr=m["lr"])
        # cosine decay (the first runs used a constant 2e-5 and were under-trained)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, tr["codec_epochs"]))
        with Timer() as t:
            for ep in range(tr["codec_epochs"]):
                codec.train()
                for x, _ in limited(train_dl, tr["max_steps_per_epoch"]):
                    pt = patches_of(x.to(dev), cfg, domain, p, m["patches_per_step"])
                    loss = F.mse_loss(codec(pt), pt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                sched.step()
                print(f"[{domain}] size {b}B epoch {ep}: train mse {loss.item():.5f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
        test_mse = evaluate(codec, test_dl, cfg, domain, p, dev)
        n_params = sum(q.numel() for q in codec.parameters())
        results[str(b)] = {"lr": m["lr"], "patches_per_step": m["patches_per_step"], "test_patch_mse": test_mse, "test_patch_psnr": float(10 * torch.log10(torch.tensor(1 / max(test_mse, 1e-10)))),
                           "params": n_params, "enc_params": sum(q.numel() for q in codec.enc.parameters()),
                           "train_seconds": t.dt, "epochs": tr["codec_epochs"],
                           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
        save_ckpt(codec, out_path(cfg, "ckpt", f"codec_{domain}_{b}.pt"))
        save_json(results, res_path)
        print(results[str(b)])


def patches_of(x_nat, cfg, domain, p, k=None):
    x = to_input(x_nat, cfg) if domain == "upsampled" else x_nat
    pt = patchify(x, p).flatten(0, 1)
    if k and pt.shape[0] > k:
        pt = pt[torch.randperm(pt.shape[0], device=pt.device)[:k]]
    return pt


@torch.no_grad()
def evaluate(codec, dl, cfg, domain, p, dev):
    codec.eval()
    s = n = 0.0
    for x, _ in limited(dl, cfg["train"]["max_steps_per_epoch"]):
        pt = patches_of(x.to(dev), cfg, domain, p)
        s += F.mse_loss(codec(pt).clamp(0, 1), pt, reduction="sum").item()
        n += pt.numel()
    return s / max(n, 1)


if __name__ == "__main__":
    main()

"""S1 - Binary (transmit-or-discard) scheme: decoder training and evaluation.

Reproduces Fig. 2 (masked MSE, accuracy vs r at alpha=1) with the corrected rate accounting, and
produces the alpha sweep with the purely random baseline alpha=0 at every rate (Fig. 4),
plus full-image PSNR / MS-SSIM and exact bits incl. the positional mask.

Decoder loss (Eq. 5): || (M kron B) . (X - X_hat) ||^2 over the transmitted patches.
Reconstructions are blackened outside the transmitted patches (as in Fig. 3) before
classification and full-image metrics.

Speed (v2): the frozen ViT is the dominant cost, so
  * all decoders (one per rate, or per (rate, alpha)) are trained TOGETHER in one pass over the data,
    sharing a single ViT forward per batch (previously: one full pass per decoder);
  * evaluation computes the ViT once per test batch and reuses it for every (rate, alpha, seed);
  * ViT and decoder forwards run in bfloat16 autocast on GPU (train.amp=true; loss in fp32).
The training objective, the selection rule, and the token quantization are unchanged."""
from _common import setup

import os
import time

import numpy as np
import pandas as pd
import torch

from semcomm import allocation as alloc
from semcomm import rate as rt
from semcomm.data import get_loader, to_input
from semcomm.metrics import masked_mse, ms_ssim, psnr
from semcomm.models import BinaryDecoder
from semcomm.system import importance, load_classifier, load_vit, mask_to_pixels
from semcomm.utils import limited, load_ckpt, out_path, save_ckpt


def extra(ap):
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--only_missing", action="store_true",
                    help="train only decoders whose checkpoint does not exist yet (e.g. new rates 0.15, 0.1)")


def quantize_tokens(z, token_bits):
    """Per-image min/max uniform quantization (b_z bits/element; 2x32-bit range sent)."""
    if not token_bits:
        return z
    lo = z.amin(dim=(1, 2), keepdim=True)
    hi = z.amax(dim=(1, 2), keepdim=True)
    L = 2 ** token_bits - 1
    return torch.round((z - lo) / (hi - lo + 1e-8) * L) / L * (hi - lo) + lo


def select(scores_np, r, alpha, rng):
    """scores_np: (B,P) numpy. Returns (B,P) bool selection (alpha-controlled attention + random fill)."""
    P = scores_np.shape[1]
    n = int(round(r * P))
    return np.stack([alloc.binary_select(s, n, alpha, rng) for s in scores_np])


def decode(dec, zq, sel, grid):
    B, P, D = zq.shape
    m = torch.from_numpy(sel).to(zq.device).unsqueeze(-1).to(zq.dtype)
    return dec((zq * m).transpose(1, 2).reshape(B, D, *grid))


def main():
    args, cfg, dev = setup(__doc__, extra)
    b, tr, p = cfg["binary"], cfg["train"], cfg["vit"]["patch"]
    H, W = cfg["data"]["input_hw"]
    grid = (H // p, W // p)
    P = grid[0] * grid[1]
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=bool(tr["amp"]) and dev.type == "cuda")
    vit = load_vit(cfg, dev)
    D = vit.backbone.embed_dim

    # which decoders to train
    if b["decoder_mode"] == "per_rate_alpha":
        jobs = [(r, (a,)) for r in b["rates"] for a in b["alpha_sweep"]]
    elif b["decoder_mode"] == "per_rate":
        jobs = [(r, tuple(b["alpha_sweep"])) for r in b["rates"]]
    else:
        jobs = [(None, tuple(b["alpha_sweep"]))]

    def ck_name(r, alphas):
        tag = "all" if r is None else f"r{r}"
        atag = f"_a{alphas[0]}" if len(alphas) == 1 else ""
        return out_path(cfg, "ckpt", f"binary_decoder_{tag}{atag}.pt")

    # ------------------------------------------------------------- training
    train_jobs = [j for j in jobs if not (args.only_missing and os.path.exists(ck_name(*j)))]
    if args.only_missing:
        print("decoders to train:", [j[0] for j in train_jobs])
    if not args.skip_train and train_jobs:
        train_dl = get_loader(cfg, "train", batch_size=b["batch_size"])
        decs, opts, rngs = [], [], []
        for j, _ in enumerate(train_jobs):
            torch.manual_seed(0)
            decs.append(BinaryDecoder(D).to(dev).train())
            opts.append(torch.optim.Adam(decs[-1].parameters(), lr=b["lr"]))
            rngs.append(np.random.default_rng(j))
        n_steps = len(train_dl) if tr["max_steps_per_epoch"] is None else min(len(train_dl), tr["max_steps_per_epoch"])
        t0 = time.time()
        for ep in range(tr["decoder_epochs"]):
            run = np.zeros(len(train_jobs))
            for it, (x, _) in enumerate(limited(train_dl, tr["max_steps_per_epoch"])):
                x_in = to_input(x.to(dev, non_blocking=True), cfg)
                with torch.no_grad(), torch.autocast(**amp):
                    z, s = importance(vit, x_in, p)  # ONE ViT forward shared by all decoders
                zq = quantize_tokens(z.float(), b["token_bits"])
                s_np = s.flatten(1).float().cpu().numpy()
                for j, ((r, alphas), dec, opt, rng) in enumerate(zip(train_jobs, decs, opts, rngs)):
                    rr = r if r is not None else float(rng.choice(b["rates"]))
                    a = float(rng.choice(alphas))
                    sel = select(s_np, rr, a, rng)
                    with torch.autocast(**amp):
                        x_hat = decode(dec, zq, sel, grid)
                    m = mask_to_pixels(sel, grid, p).to(dev)
                    loss = (((x_in - x_hat.float()) ** 2) * m).sum() / m.sum().clamp_min(1) / 3
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    run[j] += loss.item()
                if (it + 1) % 200 == 0:
                    done = ep * n_steps + it + 1
                    eta = (time.time() - t0) / done * (tr["decoder_epochs"] * n_steps - done)
                    print(f"epoch {ep} step {it + 1}/{n_steps}  ETA {eta / 3600:.1f} h  "
                          + "  ".join(f"r={jb[0]}:{run[k] / (it + 1):.5f}" for k, jb in enumerate(train_jobs)), flush=True)
            for (r, alphas), dec in zip(train_jobs, decs):
                save_ckpt(dec, ck_name(r, alphas))  # checkpoint every epoch (safe to stop early)
            print(f"epoch {ep} done, checkpoints saved", flush=True)

    if args.skip_eval:
        return
    # ------------------------------------------------------------ evaluation
    clf = load_classifier(cfg, dev)
    decs = {}
    for r, alphas in jobs:
        decs[(r, alphas)] = load_ckpt(BinaryDecoder(D), ck_name(r, alphas)).to(dev).eval()
    combos = []
    for r in b["rates"]:
        for a in b["alpha_sweep"]:
            job = [j for j in jobs if (j[0] in (r, None)) and a in j[1]][0]
            for seed in cfg["eval"]["seeds"]:
                combos.append((r, a, seed, decs[job], np.random.default_rng(1000 + seed + int(1e4 * a) + int(1e6 * r))))
    test_dl = get_loader(cfg, "test", batch_size=cfg["eval"]["batch_size"], shuffle=False)
    rows = []
    t0 = time.time()
    for it, (x, y) in enumerate(limited(test_dl, tr["max_steps_per_epoch"])):
        with torch.no_grad():
            x_in = to_input(x.to(dev), cfg)
            with torch.autocast(**amp):
                z, s = importance(vit, x_in, p)  # ONE ViT forward for all (r, alpha, seed)
            zq = quantize_tokens(z.float(), b["token_bits"])
            s_np = s.flatten(1).float().cpu().numpy()
            for r, a, seed, dec, rng in combos:
                sel = select(s_np, r, a, rng)
                with torch.autocast(**amp):
                    x_hat = decode(dec, zq, sel, grid).float()
                m = mask_to_pixels(sel, grid, p).to(dev)
                x_vis = x_hat * m  # untransmitted regions blackened (Fig. 3)
                with torch.autocast(**amp):
                    pred = clf(x_vis).argmax(1).cpu()           # untransmitted regions blackened
                    pred_raw = clf(x_hat.clamp(0, 1)).argmax(1).cpu()  # decoder output as is (no blackening)
                mm = masked_mse(x_in, x_hat, m).cpu()
                ps = psnr(x_in, x_vis).cpu()
                ss = ms_ssim(x_in, x_vis).cpu()
                for i in range(len(y)):
                    nsel = int(sel[i].sum())
                    bits = rt.binary_rate_bits(P, nsel, D, b["token_bits"]) + 64
                    rows.append(dict(r=r, alpha=a, seed=seed, correct=int(pred[i] == y[i]), correct_raw=int(pred_raw[i] == y[i]),
                                     masked_mse=float(mm[i]), psnr_full=float(ps[i]), msssim_full=float(ss[i]),
                                     bits=int(bits), bpp_native=rt.bpp(bits, cfg["data"]["native_hw"]), n_sel=nsel))
        if (it + 1) % 20 == 0:
            print(f"eval batch {it + 1}/{len(test_dl)}  {(time.time() - t0) / (it + 1) * (len(test_dl) - it - 1) / 3600:.1f} h left", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path(cfg, "results", "s1_binary_per_image.csv"), index=False)
    summ = df.groupby(["r", "alpha", "seed"]).mean(numeric_only=True).reset_index()
    summ.to_csv(out_path(cfg, "results", "s1_binary_summary.csv"), index=False)
    print(summ.groupby(["r", "alpha"]).mean(numeric_only=True)[["correct", "correct_raw", "masked_mse", "psnr_full", "bpp_native"]])


if __name__ == "__main__":
    main()

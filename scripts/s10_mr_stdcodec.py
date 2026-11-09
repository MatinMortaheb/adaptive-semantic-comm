"""S10 - attention-guided multi-resolution (MR) coding with a STANDARD back end.

The proposed MR allocation is codec-agnostic: the ViT CLS attention (computed on the 320x480 input,
pooled to the native 8x8 patch grid) ranks the native patches, and each patch is assigned a resolution
level {full, 1/2, 1/4, 1/8} by rank. The multi-resolution image is formed by blending full-image
down/up-sampled versions with smooth (bilinear) level weights, and is then coded by WebP or JPEG.
The receiver only runs the standard decoder (no side information). No training is involved.

Controls: same level mix with a RANDOM patch ranking (alpha=0 analogue), and the plain codec.
The level mix is selected on --tune_n TRAINING images (not on the test set).
Outputs:
  results/s10_per_image.csv, results/s10_summary.csv, results/s10_tuning.csv,
  results/s10_delta_at_bpp.csv   accuracy gain of MR-<codec> over <codec> at fixed bpp, bootstrap 95% CI
  results/s10_matched.csv        digital MR-<codec>/<codec> vs WITT at matched channel uses (capacity & MCS)
  paper/Figure/rev/mr_std.pdf
"""
from _common import setup

import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from semcomm import channel as ch
from semcomm import rate as rt
from semcomm.data import get_loader, resize, to_input
from semcomm.metrics import psnr
from semcomm.system import importance, load_classifier, load_vit
from semcomm.utils import limited, out_path
from semcomm.witt_adapter import channel_uses

QGRID = {"webp": [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95],
         "jpeg": [1, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]}
# fractions of patches (by importance rank) at levels [full, 1/2, 1/4, 1/8]
MIXES = {"M1": (0.75, 0.25, 0.0, 0.0), "M2": (0.5, 0.5, 0.0, 0.0), "M3": (0.5, 0.25, 0.25, 0.0),
         "M4": (0.25, 0.5, 0.25, 0.0), "M5": (0.25, 0.25, 0.5, 0.0), "M6": (0.5, 0.0, 0.25, 0.25)}
BPP_GRID = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def extra(ap):
    ap.add_argument("--codecs", nargs="*", default=["webp", "jpeg"])
    ap.add_argument("--tune_n", type=int, default=1000, help="training images used to select the level mix")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--mix", default=None, help="skip tuning and use this mix (e.g. M3)")


# ------------------------------------------------------------------ codec (runs in worker processes)
def _codec(args):
    arr, codec, q = args
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    if codec == "webp":
        im.save(buf, format="WEBP", quality=int(q), method=6)
    else:
        im.save(buf, format="JPEG", quality=int(q), subsampling=0 if q >= 90 else 2, optimize=True)
    n = buf.tell()
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB")).copy(), n


def to_u8(x):
    return (x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def from_u8(a, dev):
    return torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float().div(255).to(dev)


# ------------------------------------------------------------------ multi-resolution pre-coding
def pyramid(x, n_levels=4):
    """X_k = bicubic-up(area-down(x, 2^k)), k = 0..n_levels-1."""
    h, w = x.shape[-2:]
    out = [x]
    for k in range(1, n_levels):
        d = F.interpolate(x, size=(h >> k, w >> k), mode="area")
        out.append(F.interpolate(d, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1))
    return torch.stack(out, 1)  # (B,L,3,h,w)


def level_map(scores, mix, rng=None):
    """scores (B,gh,gw) -> one-hot level weights (B,L,gh,gw). Highest-importance patches get level 0."""
    B, gh, gw = scores.shape
    P = gh * gw
    counts = np.round(np.array(mix) * P).astype(int)
    counts[-1] = P - counts[:-1].sum()
    lv = np.repeat(np.arange(len(mix)), counts)
    out = torch.zeros(B, len(mix), P, device=scores.device)
    for b in range(B):
        order = (torch.argsort(scores[b].flatten(), descending=True) if rng is None
                 else torch.from_numpy(rng.permutation(P)).to(scores.device))
        out[b, torch.from_numpy(lv).to(scores.device), order] = 1.0
    return out.view(B, len(mix), gh, gw)


def mr_image(x, scores, mix, rng=None):
    W = level_map(scores, mix, rng)
    W = F.interpolate(W, size=x.shape[-2:], mode="bilinear", align_corners=False)
    W = W / W.sum(1, keepdim=True).clamp_min(1e-6)
    return (pyramid(x, len(mix)) * W.unsqueeze(2)).sum(1)


# ------------------------------------------------------------------ evaluation loop
def run(cfg, dev, vit, clf, dl, methods, pool, max_batches=None, tag="test"):
    """methods: list of (name, codec, mix or None, random?)"""
    h0, w0 = cfg["data"]["native_hw"]
    p = cfg["vit"]["patch"]
    rows, idx0 = [], 0
    from tqdm import tqdm
    try:
        total = max_batches or len(dl)
    except TypeError:
        total = None
    for bi, (x, y) in enumerate(tqdm(dl, total=total, desc=f"S10 {tag}")):
        if max_batches and bi >= max_batches:
            break
        x_nat = x.to(dev)
        with torch.no_grad():
            _, s = importance(vit, to_input(x_nat, cfg), p)
            s = F.adaptive_avg_pool2d(s.unsqueeze(1), (h0 // p, w0 // p)).squeeze(1)
            pre = {}
            for name, codec, mix, rnd in methods:
                key = (mix, rnd)
                if key not in pre:
                    pre[key] = x_nat if mix is None else mr_image(
                        x_nat, s, MIXES[mix], np.random.default_rng(1000 + bi) if rnd else None)
            for name, codec, mix, rnd in methods:
                src = to_u8(pre[(mix, rnd)])
                for q in QGRID[codec]:
                    res = list(pool.map(_codec, [(a, codec, q) for a in src]))
                    rec = from_u8([r[0] for r in res], dev)
                    pred = clf(to_input(rec, cfg)).argmax(1).cpu()
                    ps = psnr(x_nat, rec).cpu()
                    for i in range(len(y)):
                        rows.append(dict(img=idx0 + i, method=name, codec=codec, mix=mix or "-", q=q,
                                         bytes=res[i][1], bpp_native=rt.bpp(8 * res[i][1], (h0, w0)),
                                         correct=int(pred[i] == y[i]), psnr_native=float(ps[i])))
        idx0 += len(y)
    return pd.DataFrame(rows)


def acc_at_bpp(df, grid):
    """mean curve (over images, per q) -> accuracy interpolated in log-bpp at grid points."""
    g = df.groupby("q").agg(b=("bpp_native", "mean"), a=("correct", "mean")).sort_values("b")
    b = np.maximum.accumulate(g.b.values)  # guard tiny non-monotonicity
    return np.interp(np.log(grid), np.log(b), g.a.values, left=np.nan, right=np.nan)


def delta_ci(df, m1, m0, grid, n_boot=1000, seed=0):
    a, b = df[df.method == m1], df[df.method == m0]
    imgs = np.sort(df.img.unique())
    d0 = acc_at_bpp(a, grid) - acc_at_bpp(b, grid)
    rng = np.random.default_rng(seed)
    A = a.pivot_table(index="img", columns="q", values=["bpp_native", "correct"])
    Bp = b.pivot_table(index="img", columns="q", values=["bpp_native", "correct"])
    boots = []
    for _ in range(n_boot):
        s = rng.choice(imgs, size=len(imgs))
        def curve(T):
            bb = T["bpp_native"].loc[s].mean().values
            aa = T["correct"].loc[s].mean().values
            o = np.argsort(bb)
            return np.interp(np.log(grid), np.log(np.maximum.accumulate(bb[o])), aa[o], left=np.nan, right=np.nan)
        boots.append(curve(A) - curve(Bp))
    boots = np.array(boots)
    return d0, np.nanquantile(boots, 0.025, axis=0), np.nanquantile(boots, 0.975, axis=0)


def matched(df, cfg, methods):
    """digital transmission at the channel uses of each WITT C: per image, highest quality that fits."""
    w = cfg["witt"]
    h0, w0 = cfg["data"]["native_hw"]
    native = w["input"] == "native"
    hw0 = tuple(cfg["data"]["native_hw"] if native else cfg["data"]["input_hw"])
    snrs = list(cfg["channel"]["eval_snrs_db"])
    out = []
    for C in w["C_list"]:
        n_cu = channel_uses(C, hw0, 2 if native else 4)
        rho = rt.bandwidth_ratio(n_cu, (h0, w0))
        for snr in snrs:
            for eta_name, eta in (("capacity", float(ch.capacity_eta(snr))),
                                  ("MCS", float(ch.mcs_eta(snr, cfg["channel"]["mcs_margin_db"])))):
                budget = n_cu * eta / 8
                for m in methods:
                    d = df[(df.method == m) & (df.bytes <= budget)]
                    best = d.sort_values("q").groupby("img").tail(1)
                    n_img = df[df.method == m].img.nunique()
                    out.append(dict(method=f"{m}, {eta_name}", C=C, rho=rho, snr_db=snr, budget_bytes=budget,
                                    acc=best.correct.sum() / n_img, frac_fit=len(best) / n_img,
                                    psnr_native=best.psnr_native.mean() if len(best) else np.nan))
    return pd.DataFrame(out)


def figure(cfg, summ, mt, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = out_path(cfg, "results")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    sty = {"WebP": ("#7f7f7f", "o", "-"), "MR-WebP (proposed)": ("#1f77b4", "s", "-"),
           "MR-WebP (random)": ("#1f77b4", "s", ":"), "JPEG": ("#bcbd22", "^", "-"),
           "MR-JPEG (proposed)": ("#d62728", "D", "-"), "MR-JPEG (random)": ("#d62728", "D", ":")}
    for m, (c, mk, ls) in sty.items():
        g = summ[summ.method == m].sort_values("bpp_native")
        if len(g):
            ax[0].plot(g.bpp_native, 100 * g.correct, color=c, marker=mk, ls=ls, ms=3, label=m)
    s6 = os.path.join(res, "s6_baselines_summary.csv")
    if os.path.exists(s6):
        g = pd.read_csv(s6)
        g = g[g.method == "MR-native (proposed)"].sort_values("bpp_native")
        ax[0].plot(g.bpp_native, 100 * g.correct, color="#ff7f0e", marker="v", ls="--", ms=3, label="MR-native (learned codecs)")
    ax[0].set_xscale("log")
    ax[0].set_xlim(0.4, 8)
    ax[0].set_xlabel("rate (bits per native pixel)")
    ax[0].set_ylabel("top-1 accuracy (%)")
    ax[0].legend(fontsize=6)
    snrs = np.array(sorted(mt.snr_db.unique()))
    snr0 = float(snrs[np.argmin(abs(snrs - cfg["channel"]["awgn_snr_db"]))])
    wf = os.path.join(res, "s7_witt_per_image.csv")
    if os.path.exists(wf):
        w = pd.read_csv(wf)
        g = w[np.isclose(w.snr_db, snr0)].groupby("rho").correct.mean().reset_index()
        ax[1].plot(g.rho, 100 * g.correct, color="k", marker="D", ls="--", ms=3, label="WITT (analog JSCC)")
    for m, c, ls in ((f"MR-WebP (proposed), capacity", "#1f77b4", "-"), (f"MR-WebP (proposed), MCS", "#1f77b4", "--"),
                     ("WebP, capacity", "#7f7f7f", "-"), ("WebP, MCS", "#7f7f7f", "--")):
        g = mt[(mt.method == m) & np.isclose(mt.snr_db, snr0)].sort_values("rho")
        ax[1].plot(g.rho, 100 * g.acc, color=c, ls=ls, marker="o", ms=3, label=m)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel(f"bandwidth ratio $\\rho$ (AWGN, {snr0:g} dB)")
    ax[1].set_ylabel("top-1 accuracy (%)")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    d = out_path(cfg, "paper", "Figure", "rev")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"mr_std.{ext}"), dpi=200, bbox_inches="tight")


def main():
    args, cfg, dev = setup(__doc__, extra)
    vit, clf = load_vit(cfg, dev), load_classifier(cfg, dev)
    bs = cfg["eval"]["batch_size"]
    pool = ProcessPoolExecutor(args.workers)
    name = {"webp": "WebP", "jpeg": "JPEG"}
    os.makedirs(out_path(cfg, "results"), exist_ok=True)

    # 1) select the level mix on training images (never on test)
    chosen = {}
    if args.mix:
        chosen = {c: args.mix for c in args.codecs}
    else:
        tr = get_loader(cfg, "train", batch_size=bs, shuffle=True)
        meth = [(name[c], c, None, False) for c in args.codecs] + \
               [(f"MR-{name[c]}[{k}]", c, k, False) for c in args.codecs for k in MIXES]
        tdf = run(cfg, dev, vit, clf, tr, meth, pool, max_batches=max(1, args.tune_n // bs), tag="tuning (train)")
        trows = []
        for c in args.codecs:
            base = acc_at_bpp(tdf[tdf.method == name[c]], BPP_GRID)
            for k in MIXES:
                a = acc_at_bpp(tdf[tdf.method == f"MR-{name[c]}[{k}]"], BPP_GRID)
                trows.append(dict(codec=c, mix=k, mean_gain=np.nanmean(a - base), **{f"gain@{b}": v for b, v in zip(BPP_GRID, a - base)}))
        tune = pd.DataFrame(trows)
        tune.to_csv(out_path(cfg, "results", "s10_tuning.csv"), index=False)
        print(tune.to_string())
        for c in args.codecs:
            t = tune[tune.codec == c]
            chosen[c] = t.loc[t.mean_gain.idxmax(), "mix"]
    print("chosen level mix:", {c: (k, MIXES[k]) for c, k in chosen.items()})

    # 2) test set
    te = get_loader(cfg, "test", batch_size=bs, shuffle=False)
    meth = []
    for c in args.codecs:
        meth += [(name[c], c, None, False), (f"MR-{name[c]} (proposed)", c, chosen[c], False),
                 (f"MR-{name[c]} (random)", c, chosen[c], True)]
    df = run(cfg, dev, vit, clf, limited(te, cfg["train"]["max_steps_per_epoch"]), meth, pool, tag="test")
    df.to_csv(out_path(cfg, "results", "s10_per_image.csv"), index=False)
    summ = df.groupby(["method", "q"]).mean(numeric_only=True).reset_index()
    summ.to_csv(out_path(cfg, "results", "s10_summary.csv"), index=False)
    drows = []
    for c in args.codecs:
        for m1 in (f"MR-{name[c]} (proposed)", f"MR-{name[c]} (random)"):
            d, lo, hi = delta_ci(df, m1, name[c], BPP_GRID)
            for b, v, l_, h_ in zip(BPP_GRID, d, lo, hi):
                drows.append(dict(method=m1, vs=name[c], bpp=b, delta_acc=v, ci_lo=l_, ci_hi=h_))
    dd = pd.DataFrame(drows)
    dd.to_csv(out_path(cfg, "results", "s10_delta_at_bpp.csv"), index=False)
    pd.set_option("display.width", 200)
    print(dd.to_string())
    mt = matched(df, cfg, sorted(df.method.unique()))
    mt.to_csv(out_path(cfg, "results", "s10_matched.csv"), index=False)
    figure(cfg, summ, mt, chosen)
    pool.shutdown()


if __name__ == "__main__":
    main()

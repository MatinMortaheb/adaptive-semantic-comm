"""S3n - Multi-resolution experiments of Figs. 6 and 7 and the level ablation (Table I), at the NATIVE
TinyImageNet resolution (64x64, P=64 patches of 8x8), where the resolution levels differ in quality.
(At 320x480 an 8x8 patch of an upsampled 64x64 image covers ~1.1x1.7 native pixels, so even the
12-byte codec is nearly lossless and all levels give the same accuracy.)

Importance: ViT CLS attention on the 320x480 input, pooled to the 8x8 native grid (as MR-native).
Codecs: native codec bank (codec_native_{12,24,48}.pt + entropy_native.npz). No training.

(a) single fixed resolution (12/24/48 B, and full 192 B) on the top-q patches by attention,
    plus full resolution on a random q-fraction of patches                     -> Fig. 5, Fig. 6(a)
(b) adaptive MR with the corrected Algorithm 2 over a budget sweep          -> Fig. 6(a), 6(b)
(c) level-set ablation with the native codecs                               -> Table I
Rates include the side information (entropy-coded), bpp per native pixel. Untransmitted patches = 0.

Outputs: results/s3n_*.csv, paper/Figure/rev/{fixed_res_native,mr_accuracy_native,res_assignment_native}.pdf,
         paper/tables/level_ablation_native.tex
"""
from _common import setup

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from semcomm import rate as rt
from semcomm.data import get_loader, patchify, to_input, unpatchify
from semcomm.metrics import psnr
from semcomm.system import importance, load_classifier, load_codec_bank, load_vit, mr_transmit
from semcomm.utils import limited, load_json, out_path

Q = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
FRACS = [0.035, 0.045, 0.055, 0.0625, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]
ABL_SETS = [[0, 24, 192], [0, 12, 48, 192], [0, 12, 24, 48, 192]]
ABL_FRACS = [0.06, 0.1, 0.2]


def main():
    args, cfg, dev = setup(__doc__)
    m, p = cfg["mr"], cfg["vit"]["patch"]
    R = list(m["levels_bytes"])
    h0, w0 = cfg["data"]["native_hw"]
    gh, gw = h0 // p, w0 // p
    P = gh * gw
    R_full = P * R[-1]
    vit, clf = load_vit(cfg, dev), load_classifier(cfg, dev)
    bank = load_codec_bank(cfg, dev, domain="native")
    banks_abl = {tuple(S): load_codec_bank(cfg, dev, sizes=S, domain="native") for S in ABL_SETS}
    dl = get_loader(cfg, "test", batch_size=cfg["eval"]["batch_size"], shuffle=False)
    rng = np.random.default_rng(0)
    fixed, adapt, hist, abl = [], [], [], []

    def classify(rec_nat):
        return clf(to_input(rec_nat, cfg)).argmax(1).cpu()

    from tqdm import tqdm
    for x, y in tqdm(limited(dl, cfg["train"]["max_steps_per_epoch"]), total=len(dl), desc="S3n"):
        with torch.no_grad():
            x_nat = x.to(dev)
            B = x_nat.shape[0]
            _, s = importance(vit, to_input(x_nat, cfg), p)
            s = F.adaptive_avg_pool2d(s.unsqueeze(1), (gh, gw)).squeeze(1)
            order = s.flatten(1).argsort(1, descending=True)
            rnd = torch.stack([torch.from_numpy(rng.permutation(P)) for _ in range(B)]).to(dev)
            pt = patchify(x_nat, p)
            # (a) single fixed resolution
            for name, b, ordr in [("12 B", 12, order), ("24 B", 24, order), ("48 B", 48, order),
                                  ("full (192 B)", 192, order), ("full (192 B), random", 192, rnd)]:
                for q in Q:
                    n = int(round(q * P))
                    sizes = torch.zeros(B, P, dtype=torch.long, device=dev)
                    sizes.scatter_(1, ordr[:, :n], b)
                    rec, pbits = bank.reconstruct(pt, sizes, return_bits=True)
                    rec = unpatchify(rec, h0, w0)
                    pred = classify(rec)
                    ps = psnr(x_nat, rec).cpu()
                    bits = pbits.sum(1).cpu().numpy() + rt.binary_mask_bits(P, n) + m["header_bits"]
                    for i in range(B):
                        fixed.append(dict(scheme=name, level=b, q=q, correct=int(pred[i] == y[i]),
                                          psnr_native=float(ps[i]), bits=float(bits[i]), bpp_native=rt.bpp(bits[i], (h0, w0))))
            # (b) adaptive MR
            for f in FRACS:
                rec, lv, bits, _ = mr_transmit(bank, x_nat, s, f * R_full, R, m["header_bits"], "feasible", p)
                pred = classify(rec)
                ps = psnr(x_nat, rec).cpu()
                for i in range(B):
                    adapt.append(dict(rate_frac=f, budget_bytes=f * R_full, correct=int(pred[i] == y[i]),
                                      psnr_native=float(ps[i]), bits=float(bits[i]), bpp_native=rt.bpp(bits[i], (h0, w0))))
                    hist.append(dict(rate_frac=f, bpp_native=rt.bpp(bits[i], (h0, w0)),
                                     **{f"L{R[k]}": float((lv[i] == k).mean()) for k in range(len(R))}))
            # (c) level-set ablation
            for S, bk in banks_abl.items():
                for f in ABL_FRACS:
                    rec, lv, bits, _ = mr_transmit(bk, x_nat, s, f * R_full, list(S), m["header_bits"], "feasible", p)
                    pred = classify(rec)
                    for i in range(B):
                        abl.append(dict(levels="{" + ",".join(map(str, S)) + "}", rate_frac=f, correct=int(pred[i] == y[i]),
                                        bpp_native=rt.bpp(bits[i], (h0, w0))))
    res = out_path(cfg, "results")
    fx, ad, hs, ab = map(pd.DataFrame, (fixed, adapt, hist, abl))
    fx.to_csv(f"{res}/s3n_fixed_native.csv", index=False)
    ad.to_csv(f"{res}/s3n_adaptive_native.csv", index=False)
    hs = hs.groupby("rate_frac").mean().reset_index()
    hs.to_csv(f"{res}/s3n_level_hist_native.csv", index=False)
    ab.to_csv(f"{res}/s3n_ablation_native.csv", index=False)
    pd.set_option("display.width", 200)
    print(fx.groupby(["scheme", "q"])[["correct", "psnr_native", "bpp_native"]].mean().round(3).to_string())
    print(ad.groupby("rate_frac")[["correct", "psnr_native", "bpp_native"]].mean().round(3).to_string())
    print(ab.groupby(["levels", "rate_frac"])[["correct", "bpp_native"]].mean().round(3).to_string())
    figures(cfg, fx, ad, hs, ab, R)


def figures(cfg, fx, ad, hs, ab, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    figdir = out_path(cfg, "paper", "Figure", "rev")
    tabdir = out_path(cfg, "paper", "tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)
    sty = {"12 B": ("#eda100", "^", "-."), "24 B": ("#1baf7a", "D", "--"), "48 B": ("#6250d6", "v", ":"),
           "full (192 B)": ("#2a78d6", "o", "-"), "full (192 B), random": ("#e34948", "x", "-")}

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(figdir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    g = fx.groupby(["scheme", "q"]).mean(numeric_only=True).reset_index()
    # Fig. 5: single resolution vs q (attention selection)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for sch in ["12 B", "24 B", "48 B", "full (192 B)"]:
        c, mk, ls = sty[sch]
        d = g[g.scheme == sch]
        dp = d[d.psnr_native < 99]  # full resolution at q=1 is lossless (infinite PSNR): not plotted
        ax[0].plot(dp.q, dp.psnr_native, color=c, marker=mk, ls=ls, ms=3.5, label=sch)
        ax[1].plot(d.q, 100 * d.correct, color=c, marker=mk, ls=ls, ms=3.5, label=sch)
    ax[0].annotate("full resolution at $q$=1:\nlossless", xy=(0.04, 0.96), xycoords="axes fraction",
                   ha="left", va="top", fontsize=6, color=sty["full (192 B)"][0])
    ax[0].set_ylabel("PSNR (dB)")
    ax[1].set_ylabel("top-1 accuracy (%)")
    for a, t in zip(ax, ("(a)", "(b)")):
        a.set_xlabel("fraction of patches transmitted $q$")
        a.set_title(t, fontsize=8)
    ax[1].legend(fontsize=6.5)
    save(fig, "fixed_res_native")

    # Fig. 6(a): accuracy vs rate
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    a = ad.groupby("rate_frac").mean(numeric_only=True).reset_index()
    ax.plot(a.bpp_native, 100 * a.correct, color="#0b0b0b", marker="s", lw=2, ms=4, label="multi-resolution (Alg. 2)")
    for sch in ["12 B", "24 B", "48 B", "full (192 B)", "full (192 B), random"]:
        c, mk, ls = sty[sch]
        d = g[g.scheme == sch].sort_values("bpp_native")
        ax.plot(d.bpp_native, 100 * d.correct, color=c, marker=mk, ls=ls, ms=3,
                label=("single resolution " + sch) if "random" not in sch else "full resolution, random selection")
    ax.set_xscale("log")
    ax.set_xlabel("rate (bits per native pixel, incl. side info)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.legend(fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.2), frameon=False)
    save(fig, "mr_accuracy_native")

    # Fig. 6(b): resolution assignment
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    cols = ["#d9d9d9", "#eda100", "#1baf7a", "#6250d6", "#2a78d6"]
    x = np.arange(len(hs))
    bottom = np.zeros(len(hs))
    for k, b in enumerate(R):
        v = 100 * hs[f"L{b}"].values
        ax.bar(x, v, bottom=bottom, color=cols[k % len(cols)], label=f"{b} B" if b else "0 B (not sent)")
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in hs.bpp_native], rotation=60, fontsize=6)
    ax.set_xlabel("rate (bits per native pixel)")
    ax.set_ylabel("patches (%)")
    ax.legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(False)
    save(fig, "res_assignment_native")

    # Table I (native): accuracy at three budgets + codec overhead
    info = {}
    jf = out_path(cfg, "results", "s2_codecs_native.json")
    if os.path.exists(jf):
        info = load_json(jf)
    t = ab.groupby(["levels", "rate_frac"]).agg(acc=("correct", "mean"), bpp=("bpp_native", "mean")).reset_index()
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Levels (B) & \multicolumn{3}{c}{Acc. (\%) at budget (bpp)} & Params & Train \\",
             " & " + " & ".join(f"{f * 24:.1f}" for f in ABL_FRACS) + r" & (M) & (GPU-h) \\", r"\midrule"]
    for S in ABL_SETS:
        key = "{" + ",".join(map(str, S)) + "}"
        accs = [t[(t.levels == key) & np.isclose(t.rate_frac, f)].acc.mean() for f in ABL_FRACS]
        mids = [b for b in S if b not in (0, 192)]
        par = sum(info.get(str(b), {}).get("params", np.nan) for b in mids) / 1e6
        hrs = sum(info.get(str(b), {}).get("train_seconds", np.nan) for b in mids) / 3600
        lab = r"$\{" + ",".join(map(str, S)) + r"\}$" + (" (default)" if S == ABL_SETS[-1] else "")
        lines.append(lab + " & " + " & ".join(f"{100 * v:.1f}" for v in accs) + f" & {par:.1f} & {hrs:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tabdir, "level_ablation_native.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote figures and table")


if __name__ == "__main__":
    main()

"""Build every figure (PDF+PNG) and LaTeX table from outputs/results/*.csv.

Output layout matches the Overleaf project:
  outputs/paper/Figure/rev/<name>.pdf   -> copy to  <overleaf>/Figure/rev/
  outputs/paper/tables/<name>.tex       -> copy to  <overleaf>/tables/
The revised manuscript includes these files automatically when present (\\IfFileExists),
and shows a red placeholder otherwise. Missing inputs are skipped with a message."""
from _common import setup

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from semcomm.metrics import bootstrap_ci  # noqa: E402

# Categorical slots in fixed order (validated reference palette); markers/line styles as secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LS = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
INK, MUTED = "#0b0b0b", "#8a8984"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.4, "lines.markersize": 4,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, cfg, name):
    d = os.path.join(cfg["out_dir"], "paper", "Figure", "rev")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def write_table(cfg, name, body):
    d = os.path.join(cfg["out_dir"], "paper", "tables")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".tex"), "w") as f:
        f.write(body)
    print("wrote table", name)


def res(cfg, fn):
    p = os.path.join(cfg["out_dir"], "results", fn)
    if not os.path.exists(p):
        print("[skip] missing", p)
        return None
    return pd.read_csv(p)


def ci_group(df, keys, col, n_boot):
    out = []
    for k, g in df.groupby(keys):
        m, lo, hi = bootstrap_ci(g[col].values, n=n_boot)
        out.append(dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else (k,)), mean=m, lo=lo, hi=hi))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures
def fig_alpha_sweep(cfg):
    """(a) receiver uses the decoder output everywhere ('correct_raw');
    (b) untransmitted regions blackened before classification ('correct'). Mean and 95% CI."""
    d = res(cfg, "s1_binary_per_image.csv")
    if d is None:
        return
    nb = cfg["eval"]["bootstrap"]
    cols = [("correct_raw", "(a) decoder output"), ("correct", "(b) untransmitted regions blackened")]
    cols = [c for c in cols if c[0] in d]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 2.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (col, title) in zip(axes, cols):
        s = ci_group(d, ["alpha", "r"], col, nb)
        for i, (a, g) in enumerate(s.groupby("alpha")):
            g = g.sort_values("r")
            lab = f"$\\alpha$={a:g}" + (" (random)" if a == 0 else "")
            ax.errorbar(g.r, 100 * g["mean"], yerr=[100 * (g["mean"] - g.lo), 100 * (g.hi - g["mean"])],
                        color=C[i % 8], marker=MK[i % 8], ls=LS[i % 8], capsize=2, label=lab)
        ax.set_title(title)
        ax.set_xlabel("normalized rate $r$ (fraction of patches)")
        ax.set_ylabel("top-1 accuracy (%)")
    axes[0].legend(ncol=6 if len(cols) > 1 else 3, loc="upper center", bbox_to_anchor=(0.5 * len(cols) + 0.05 * (len(cols) - 1), -0.22))
    save(fig, cfg, "alpha_sweep")
    # Fig. 2 (binary scheme, alpha=1): (a) masked MSE over the transmitted patches,
    # (b) full-image PSNR and MS-SSIM (untransmitted patches set to zero)
    g1 = d[d.alpha == cfg["binary"]["alpha_main"]].groupby("r").mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax[0].plot(g1.r, g1.masked_mse, color=C[0], marker=MK[0])
    ax[0].set_xlabel("normalized rate $r$")
    ax[0].set_ylabel("masked MSE (transmitted patches)")
    ax[0].set_title("(a)", fontsize=8)
    l1 = ax[1].plot(g1.r, g1.psnr_full, color=C[1], marker=MK[1], label="PSNR")
    ax[1].set_xlabel("normalized rate $r$")
    ax[1].set_ylabel("full-image PSNR (dB)")
    ax[1].set_title("(b)", fontsize=8)
    if "msssim_full" in g1:
        a2 = ax[1].twinx()
        a2.spines["right"].set_visible(True)
        a2.grid(False)
        l2 = a2.plot(g1.r, g1.msssim_full, color=C[2], marker=MK[2], ls="--", label="MS-SSIM")
        a2.set_ylabel("full-image MS-SSIM")
        a2.set_ylim(0, 1)
        ax[1].legend(l1 + l2, [l.get_label() for l in l1 + l2], loc="upper left", fontsize=7)
    save(fig, cfg, "binary_mse_psnr")


def fig_mr(cfg):
    a, fx, h = res(cfg, "s3_adaptive.csv"), res(cfg, "s3_fixed_res.csv"), res(cfg, "s3_level_hist.csv")
    if a is not None:
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        g = a.groupby("rate_frac").mean(numeric_only=True).reset_index()
        ax.plot(g.bpp_native, 100 * g.correct, color=C[0], marker=MK[0], label="adaptive MR (corrected Alg. 2)")
        if fx is not None:
            for i, (b, gb) in enumerate(fx.groupby("level")):
                gb = gb.groupby("q").mean(numeric_only=True).reset_index()
                ax.plot(gb.bpp_native, 100 * gb.correct, color=C[i + 1], marker=MK[i + 1], ls=LS[i + 1],
                        label=f"single resolution {b} B")
        ax.set_xscale("log")
        ax.set_xlabel("rate (bits per native pixel, incl. side info)")
        ax.set_ylabel("top-1 accuracy (%)")
        ax.legend()
        save(fig, cfg, "mr_accuracy")
    if h is not None:
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        R = cfg["mr"]["levels_bytes"]
        bottom = np.zeros(len(h))
        x = np.arange(len(h))
        for k in range(len(R)):
            v = 100 * h[f"L{k}"].values
            ax.bar(x, v, bottom=bottom, color=(["#d9d8d3"] + C)[k], width=0.8,
                   edgecolor="white", linewidth=1, label=f"{R[k]} B")
            bottom += v
        ax.set_xticks(x, [f"{f:g}" for f in h.rate_frac], rotation=45)
        ax.set_xlabel("budget / $R_{\\mathrm{full}}$")
        ax.set_ylabel("patches (%)")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        ax.grid(axis="x", visible=False)
        save(fig, cfg, "res_assignment")


def fig_baselines(cfg):
    """explicit styles/labels; WITT is compared in fig_witt only (it has no bitstream)."""
    b = res(cfg, "s6_baselines_summary.csv")
    if b is None:
        return
    s1 = res(cfg, "s1_binary_summary.csv")
    # method -> (label, colour, marker, linestyle, emphasised)
    sty = {
        "MR (proposed)": ("MR (proposed), upsampled", "#2a78d6", "o", "-", True),
        "MR-native (proposed)": ("MR-native (proposed)", "#e34948", "s", "-", True),
        "Uniform": ("Uniform-rate, upsampled", "#2a78d6", "o", ":", False),
        "Uniform-native": ("Uniform-rate, native", "#e34948", "s", ":", False),
        "IAQ [Park]": ("IAQ [13], upsampled", "#6250d6", "v", "--", False),
        "IAQ-native [Park]": ("IAQ [13], native", "#008300", "v", "--", False),
        "JPEG": ("JPEG, upsampled image", "#eda100", "^", "-.", False),
        "JP2": ("JPEG2000, upsampled image", "#e87ba4", "D", "-.", False),
        "WEBP": ("WebP, upsampled image", "#1baf7a", "P", "-.", False),
        "Native-JPEG": ("JPEG, native image", "#eda100", "^", "-", False),
        "Native-JP2": ("JPEG2000, native image", "#e87ba4", "D", "-", False),
        "Native-WEBP": ("WebP, native image", "#1baf7a", "P", "-", False),
    }
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for mth, (lab, col, mk, ls, emph) in sty.items():
        if mth not in set(b.method):
            continue
        g = b[b.method == mth].sort_values("bpp_native")
        kw = dict(color=col, marker=mk, ls=ls, label=lab, ms=4 if emph else 2.5, lw=2.0 if emph else 1.0)
        ax[0].plot(g.bpp_native, 100 * g.correct, **kw)
        ax[1].plot(g.bpp_native, g.psnr_native, **kw)
    if s1 is not None and "correct_raw" in s1:
        g = s1[s1.alpha == cfg["binary"]["alpha_main"]].groupby("r").mean(numeric_only=True).reset_index()
        ax[0].plot(g.bpp_native, 100 * g.correct_raw, color=INK, marker="x", ls=":", ms=3.5,
                   label="Binary scheme ($\\alpha$=1, $r$=0.25-1)")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("rate (bits per native pixel, incl. side info)")
        a_.axvline(24, color=MUTED, lw=0.8, ls="--")
    ax[0].set_ylabel("top-1 accuracy (%)")
    ax[1].set_ylabel("PSNR at native resolution (dB)")
    ax[0].text(24, ax[0].get_ylim()[0], " raw native image", color=MUTED, fontsize=6, va="bottom")
    ax[0].set_title("(a)", fontsize=8)
    ax[1].set_title("(b)", fontsize=8)
    h, l = ax[0].get_legend_handles_labels()
    ax[1].legend(h, l, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
    save(fig, cfg, "baselines")


def fig_witt(cfg):
    w, dg = res(cfg, "s7_witt_per_image.csv"), res(cfg, "s7_digital_matched_per_image.csv")
    if w is None:
        return
    allr = pd.concat([w, dg]) if dg is not None else w
    snrs = np.array(sorted(allr.snr_db.unique()))
    snr0 = float(snrs[np.argmin(abs(snrs - cfg["channel"]["awgn_snr_db"]))])  # nearest evaluated SNR
    rhos = sorted(allr.rho.unique())
    rho_mid = 0.25 if any(np.isclose(rhos, 0.25)) else rhos[len(rhos) // 2]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
    order = ["WITT", "MR-native (proposed), capacity", "MR-native (proposed), MCS",
             "MR (proposed), capacity", "MR (proposed), MCS"]
    for i, mth in enumerate([m for m in order if m in set(allr.method)]):
        g = allr[allr.method == mth]
        g1 = g[np.isclose(g.snr_db, snr0)].groupby("rho").correct.mean().reset_index()
        ax[0].plot(g1.rho, 100 * g1.correct, color=C[i], marker=MK[i], ls=LS[i], label=mth)
        g2 = g[np.isclose(g.rho, rho_mid)].groupby("snr_db").correct.mean().reset_index()
        ax[1].plot(g2.snr_db, 100 * g2.correct, color=C[i], marker=MK[i], ls=LS[i], label=mth)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel(f"bandwidth ratio $\\rho$ (AWGN, {snr0:g} dB)")
    ax[1].set_xlabel(f"SNR (dB), $\\rho$={rho_mid:g}")
    for a_ in ax:
        a_.set_ylabel("top-1 accuracy (%)")
    ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, cfg, "witt_matched")


def fig_robust(cfg):
    """(a),(b): fading at each bandwidth ratio, perfect (solid) vs sigma=0.5 (dashed) rate information;
    (c): packet erasures (curves labelled by rho). One shared legend below the panels.
    Upsampled-domain schemes are omitted (budgets here are far below one 12-byte level per patch)."""
    from matplotlib.lines import Line2D
    f, e = res(cfg, "s8_fading_per_image.csv"), res(cfg, "s8_erasure_per_image.csv")
    if f is None:
        return
    keep = ["MR-native (proposed)", "Uniform-native"]
    name = {"MR-native (proposed)": "MR-native (proposed)", "Uniform-native": "uniform-rate coding"}
    col = {"MR-native (proposed)": C[0], "Uniform-native": C[1]}
    mk = {"MR-native (proposed)": "o", "Uniform-native": "s"}
    rhos = sorted(f.rho.unique())
    sigs = sorted(f.csi_sigma.unique())
    sig_pair = [sigs[0], sigs[-1]]
    n = len(rhos) + (e is not None)
    fig, ax = plt.subplots(1, n, figsize=(2.35 * n, 2.5))
    letters = "abcd"
    for k, rho in enumerate(rhos):
        for m in keep:
            for j, sig in enumerate(sig_pair):
                g = f[(f.method == m) & np.isclose(f.rho, rho) & np.isclose(f.csi_sigma, sig)]
                if len(g) == 0:
                    continue
                gg = g.groupby("avg_snr_db").correct.mean().reset_index()
                ax[k].plot(gg.avg_snr_db, 100 * gg.correct, color=col[m], marker=mk[m], ms=3.5,
                           ls="-" if j == 0 else "--", mfc=col[m] if j == 0 else "white")
        ax[k].set_title(f"({letters[k]}) block fading, $\\rho$={rho:g}", fontsize=8)
        ax[k].set_xlabel("average SNR (dB)")
        ax[k].set_ylabel("top-1 accuracy (%)")
    if e is not None:
        a = ax[-1]
        for m in keep:
            for rho in rhos:
                gg = e[(e.method == m) & np.isclose(e.rho, rho)].groupby("p_e").correct.mean().reset_index()
                a.plot(gg.p_e, 100 * gg.correct, color=col[m], marker=mk[m], ms=3.5, ls="-")
        for rho in rhos:
            gg = e[(e.method == keep[0]) & np.isclose(e.rho, rho)].groupby("p_e").correct.mean().reset_index()
            a.annotate(f"$\\rho$={rho:g}", xy=(gg.p_e.iloc[-1], 100 * gg.correct.iloc[-1]), xytext=(-4, 7),
                       textcoords="offset points", ha="right", fontsize=7)
        a.set_title(f"({letters[len(rhos)]}) packet erasures", fontsize=8)
        a.set_xlabel("erasure probability $p_e$")
        a.set_ylabel("top-1 accuracy (%)")
    handles = []
    for m in keep:  # one entry per drawn curve style
        handles.append(Line2D([], [], color=col[m], marker=mk[m], ms=4, ls="-", label=name[m]))
        handles.append(Line2D([], [], color=col[m], marker=mk[m], ms=4, ls="--", mfc="white",
                              label=f"{name[m]}, imperfect rate information ($\\sigma_\\epsilon$={sig_pair[-1]:g})"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.03),
               handlelength=3)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save(fig, cfg, "robust")


# ------------------------------------------------------------------ tables
def tables(cfg):
    lv = res(cfg, "s4_level_ablation.csv")
    if lv is not None:
        fr = cfg["mr"]["eval_rate_fracs_ablation"]
        head = " & ".join([f"{100*f:g}\\%" for f in fr])
        body = "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}l" + "c" * len(fr) + "cc@{}}\n\\toprule\n"
        body += "Levels (B) & \\multicolumn{" + str(len(fr)) + "}{c}{Acc. (\\%) at budget/$R_{\\mathrm{full}}$} & Params & Train \\\\\n"
        body += " & " + head + " & (M) & (GPU-h) \\\\\n\\midrule\n"
        for _, r in lv.iterrows():
            accs = " & ".join(f"{100*r[f'acc@{f}']:.1f}" for f in fr)
            lev = r.levels.replace("{", "\\{").replace("}", "\\}")
            body += f"${lev}$ & {accs} & {r.codec_params_K/1e3:.1f} & {r.train_gpu_hours:.2f} \\\\\n"
        body += "\\bottomrule\n\\end{tabular}\n"
        write_table(cfg, "level_ablation", body)
    rt_ = res(cfg, "s5_rate_table.csv")
    if rt_ is not None:
        body = "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nOperating point & Payload & Side info & Total & Side & bpp \\\\\n & (B) & (B) & (B) & (\\%) & (native) \\\\\n\\midrule\n"
        for _, r in rt_.iterrows():
            op = r.operating_point.replace("R_full", "$R_{\\mathrm{full}}$") if "$" not in r.operating_point else r.operating_point
            body += f"{r.scheme}, {op} & {r.payload_bytes:,.0f} & {r.side_bytes:,.0f} & {r.total_bytes:,.0f} & {r.side_pct:.2f} & {r.bpp_native:.1f} \\\\\n".replace(",", "{,}")
        body += "\\bottomrule\n\\end{tabular}\n"
        write_table(cfg, "rate_accounting", body)
    ov = res(cfg, "s3_alg2_overshoot.csv")
    if ov is not None:
        body = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nBudget/$R_{\\mathrm{full}}$ & Over budget (\\%) & Mean excess (B) & Max excess (B) \\\\\n\\midrule\n"
        for _, r in ov.iterrows():
            body += f"{r.rate_frac:g} & {100*r.frac_images_over_budget:.1f} & {r.mean_over_bytes:.1f} & {r.max_over_bytes:.0f} \\\\\n"
        body += "\\bottomrule\n\\end{tabular}\n"
        write_table(cfg, "alg2_overshoot", body)
    cx = res(cfg, "s9_complexity.csv")
    if cx is not None:
        body = "\\begin{tabular}{@{}p{3.6cm}ccc@{}}\n\\toprule\nComponent & Params (M) & GFLOPs & Latency (ms) \\\\\n\\midrule\n"
        for _, r in cx.iterrows():
            gf = "--" if pd.isna(r.gflops) else f"{r.gflops:.2f}"
            la = "--" if pd.isna(r.latency_ms) else f"{r.latency_ms:.2f}"
            body += f"{r.component} & {r.params_M:.3f} & {gf} & {la} \\\\\n"
        body += f"\\bottomrule\n\\multicolumn{{4}}{{l}}{{\\footnotesize Device: {cx.device.iloc[0]}}}\\\\\n\\end{{tabular}}\n"
        write_table(cfg, "complexity", body)


def main():
    args, cfg, dev = setup(__doc__)
    for fn in (fig_alpha_sweep, fig_mr, fig_baselines, fig_witt, fig_robust, tables):
        try:
            fn(cfg)
        except Exception as ex:  # keep going: one broken input must not block the others
            print(f"[error] {fn.__name__}: {ex}")


if __name__ == "__main__":
    main()

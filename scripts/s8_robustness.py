"""S8 - Robustness.

(a) Rayleigh block-fading rate trace (one image per block), MCS-table budgets, with perfect and imperfect
    SNR knowledge. With overestimated SNR, delivery is truncated ('truncate') or the block is lost
    ('drop_block'); MR packets are sent in decreasing importance order (side info first), the uniform
    baseline in raster order. WITT sees the true SNR on the channel and the estimated SNR in its ModNet.
(b) Independent packet erasures at a fixed budget (side information assumed protected)."""
from _common import setup

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from semcomm import allocation as alloc
from semcomm import channel as ch
from semcomm import rate as rt
from semcomm.data import get_loader, patchify, resize, to_input, unpatchify
from semcomm.metrics import psnr
from semcomm.system import importance, load_classifier, load_codec_bank, load_vit
from semcomm.utils import limited, load_ckpt, out_path


def extra(ap):
    ap.add_argument("--witt_C", type=int, default=None, help="WITT model (C) to use in the fading test")


def code_all(bank, x_code, levels, R, p):
    """Encode one image with the given levels. Returns per-patch reconstructions (P,3,p,p) and the
    per-patch payload in bytes (entropy-coded if the bank has an entropy model)."""
    sizes = torch.as_tensor(np.asarray(R)[levels], device=x_code.device).unsqueeze(0)
    rec, pbits = bank.reconstruct(patchify(x_code, p), sizes, return_bits=True)
    return rec[0], pbits[0].cpu().numpy() / 8.0


def assemble(rec_patches, keep, H, W):
    k = torch.as_tensor(keep, device=rec_patches.device, dtype=rec_patches.dtype).view(-1, 1, 1, 1)
    return unpatchify((rec_patches * k).unsqueeze(0), H, W)


def keep_after_truncation(patch_bytes, order, delivered_bytes, side_bytes):
    keep = np.zeros(len(patch_bytes), dtype=bool)
    avail = delivered_bytes - side_bytes
    if avail < 0:
        return keep
    cum = np.cumsum(patch_bytes[order])
    keep[order[cum <= avail + 1e-9]] = True
    return keep


def fill_lost(rec, keep, p, mode):
    if mode != "mean" or keep.all() or not keep.any():
        return rec
    B, C, H, W = rec.shape
    k = torch.as_tensor(keep, device=rec.device).view(1, 1, H // p, W // p).float()
    kp = F.interpolate(k, scale_factor=p, mode="nearest")
    mean = (rec * kp).sum((2, 3), keepdim=True) / kp.sum((2, 3), keepdim=True).clamp_min(1)
    return rec * kp + mean * (1 - kp)


def main():
    args, cfg, dev = setup(__doc__, extra)
    c, m, p = cfg["channel"], cfg["mr"], cfg["vit"]["patch"]
    R = list(m["levels_bytes"])
    h0, w0 = cfg["data"]["native_hw"]
    vit, clf = load_vit(cfg, dev), load_classifier(cfg, dev)
    banks = {"MR (proposed)": ("upsampled", load_codec_bank(cfg, dev, domain="upsampled")),
             "MR-native (proposed)": ("native", load_codec_bank(cfg, dev, domain="native"))}
    witt = None
    if args.witt_C is not None:
        from semcomm.witt_adapter import build_witt
        witt, _ = build_witt(cfg, args.witt_C, dev)
        load_ckpt(witt, out_path(cfg, "ckpt", f"witt_{cfg['witt']['input']}_C{args.witt_C}.pt"))
        witt.eval()
    test_dl = get_loader(cfg, "test", batch_size=1, shuffle=False)
    _cache = {}

    def imp_cache(t, x_in):  # test image t is revisited by every trace; run the ViT only once per image
        if t not in _cache:
            _cache[t] = importance(vit, x_in, p)[1]
        return _cache[t]

    rows, erows = [], []
    rng = np.random.default_rng(0)
    for rho in c["bandwidth_ratios"]:
        n_cu = rho * 3 * h0 * w0
        for avg in c["avg_snrs_db"]:
            for sig in c["csi_sigma"]:
                rng_t = np.random.default_rng(int(1000 * avg + 100 * sig))
                for t, (x, y) in enumerate(limited(test_dl, min(c["n_blocks"], cfg["data"]["num_test"] or 10 ** 9))):
                    g = float(ch.rayleigh_trace(1, avg, rng_t)[0])
                    gh = float(ch.estimated_snr_db(np.array([g]), sig, rng_t)[0])
                    planned, delivered = ch.block_budget_bits(n_cu, g, gh, c["mcs_margin_db"], c["outage_model"])
                    with torch.no_grad():
                        x_nat = x.to(dev)
                        x_in = to_input(x_nat, cfg)
                        s = imp_cache(t, x_in)
                        for name, (dom, bank) in banks.items():
                            xc = x_in if dom == "upsampled" else x_nat
                            gh_, gw_ = xc.shape[-2] // p, xc.shape[-1] // p
                            sc = F.adaptive_avg_pool2d(s.unsqueeze(1), (gh_, gw_)).flatten().double().cpu().numpy()
                            P = gh_ * gw_
                            side = rt.mr_side_bits_fixed(P, len(R), m["header_bits"]) / 8
                            costs = bank.level_costs(R)
                            H_, W_ = xc.shape[-2:]
                            # proposed: allocation on the PLANNED budget, importance-ordered packets
                            lv = alloc.alg2_feasible(sc, planned / 8, costs, side_bytes=side)
                            rp, pb = code_all(bank, xc, lv, R, p)
                            keep = keep_after_truncation(pb, np.argsort(-sc, kind="stable"), delivered / 8, side)
                            # uniform baseline: raster-ordered packets
                            lu = alloc.uniform_levels(P, planned / 8 - 2, costs, rng)
                            ru, pbu = code_all(bank, xc, lu, R, p)
                            keep_u = keep_after_truncation(pbu, np.arange(P), delivered / 8, 2)
                            for meth, rpatch, kp in ((name, rp, keep), (name.replace("MR", "Uniform").replace(" (proposed)", ""), ru, keep_u)):
                                rec = assemble(rpatch, kp, H_, W_)
                                rec_in = rec if dom == "upsampled" else to_input(rec, cfg)
                                rows.append(dict(method=meth, rho=rho, avg_snr_db=avg, csi_sigma=sig, snr_db=g, snr_hat_db=gh,
                                                 planned_bits=planned, delivered_bits=delivered,
                                                 correct=int(clf(rec_in).argmax(1).item() == y.item())))
                        if witt is not None and abs(rho - cfg_rho(cfg, args.witt_C)) < 1e-6:
                            xin = x_nat if cfg["witt"]["input"] == "native" else x_in
                            rec = witt_fading(witt, xin, g, gh)
                            rec_in = to_input(rec, cfg) if cfg["witt"]["input"] == "native" else rec
                            rows.append(dict(method="WITT", rho=rho, avg_snr_db=avg, csi_sigma=sig, snr_db=g, snr_hat_db=gh,
                                             planned_bits=np.nan, delivered_bits=np.nan,
                                             correct=int(clf(rec_in).argmax(1).item() == y.item())))
        # (b) erasures at a fixed budget = mean MCS rate at the middle average SNR
        budget_bits = n_cu * float(np.mean(ch.mcs_eta(ch.rayleigh_trace(10000, c["avg_snrs_db"][len(c["avg_snrs_db"]) // 2], rng), c["mcs_margin_db"])))
        for pe in c["erasure_probs"]:
            rng_e = np.random.default_rng(int(1e4 * pe) + 7)
            for t, (x, y) in enumerate(limited(test_dl, min(c["n_blocks"], cfg["data"]["num_test"] or 10 ** 9))):
                with torch.no_grad():
                    x_nat = x.to(dev)
                    x_in = to_input(x_nat, cfg)
                    s = imp_cache(t, x_in)
                    for name, (dom, bank) in banks.items():
                        xc = x_in if dom == "upsampled" else x_nat
                        gh_, gw_ = xc.shape[-2] // p, xc.shape[-1] // p
                        sc = F.adaptive_avg_pool2d(s.unsqueeze(1), (gh_, gw_)).flatten().double().cpu().numpy()
                        P = gh_ * gw_
                        side = rt.mr_side_bits_fixed(P, len(R), m["header_bits"]) / 8
                        costs = bank.level_costs(R)
                        H_, W_ = xc.shape[-2:]
                        lv = alloc.alg2_feasible(sc, budget_bits / 8, costs, side_bytes=side)
                        lu = alloc.uniform_levels(P, budget_bits / 8 - 2, costs, rng)
                        for meth, lev in ((name, lv), (name.replace("MR", "Uniform").replace(" (proposed)", ""), lu)):
                            sent = np.nonzero(lev > 0)[0]
                            n_pk = int(np.ceil(len(sent) / c["patches_per_packet"]))
                            ok = ch.packet_erasure_mask(n_pk, pe, rng_e)
                            keep = np.zeros(P, dtype=bool)
                            for k in range(n_pk):
                                if ok[k]:
                                    keep[sent[k * c["patches_per_packet"]:(k + 1) * c["patches_per_packet"]]] = True
                            rpatch, _ = code_all(bank, xc, lev, R, p)
                            rec = assemble(rpatch, keep | (lev == 0), H_, W_)
                            rec = fill_lost(rec, keep | (lev == 0), p, c["erasure_fill"])
                            rec_in = rec if dom == "upsampled" else to_input(rec, cfg)
                            erows.append(dict(method=meth, rho=rho, p_e=pe, budget_bits=budget_bits,
                                              correct=int(clf(rec_in).argmax(1).item() == y.item())))
    pd.DataFrame(rows).to_csv(out_path(cfg, "results", "s8_fading_per_image.csv"), index=False)
    pd.DataFrame(erows).to_csv(out_path(cfg, "results", "s8_erasure_per_image.csv"), index=False)
    print(pd.DataFrame(rows).groupby(["method", "rho", "avg_snr_db", "csi_sigma"]).correct.mean())
    print(pd.DataFrame(erows).groupby(["method", "rho", "p_e"]).correct.mean())


def cfg_rho(cfg, C):
    from semcomm.witt_adapter import channel_uses
    native = cfg["witt"]["input"] == "native"
    hw = cfg["data"]["native_hw"] if native else cfg["data"]["input_hw"]
    return rt.bandwidth_ratio(channel_uses(C, hw, 2 if native else 4), cfg["data"]["native_hw"])


@torch.no_grad()
def witt_fading(net, x, snr_true, snr_hat):
    B, _, H, W = x.shape
    if H != net.H or W != net.W:
        net(x, given_SNR=snr_hat)  # sets the resolution of encoder/decoder
    feat = net.encoder(x, snr_hat, net.model)
    noisy = net.feature_pass_channel(feat, snr_true)
    return net.decoder(noisy, snr_hat, net.model).clamp(0, 1)


if __name__ == "__main__":
    main()

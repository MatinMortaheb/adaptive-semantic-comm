"""S7 - WITT baseline and matched-channel-use comparison with the proposed digital scheme.

1) trains WITT (authors' code, random-SNR training with ModNet) for each bottleneck C in witt.C_list
2) evaluates WITT over AWGN at channel.eval_snrs_db: PSNR (native) and accuracy of the same classifier
3) evaluates the proposed MR schemes at the SAME number of channel uses n_cu and SNR, with
   bit budget n_cu * eta, eta = log2(1+SNR) (capacity, favours ours) and eta = MCS table (practical)."""
from _common import setup

import os

import numpy as np
import pandas as pd
import torch

from semcomm import channel as ch
from semcomm import rate as rt
from semcomm.data import get_loader, resize, to_input
from semcomm.metrics import psnr
from semcomm.system import importance, load_classifier, load_codec_bank, load_vit, mr_transmit
from semcomm.utils import limited, load_ckpt, out_path, save_ckpt
from semcomm.witt_adapter import build_witt, channel_uses


def extra(ap):
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_digital", action="store_true")
    ap.add_argument("--digital_only", action="store_true",
                    help="skip WITT training/eval and reuse results/s7_witt_per_image.csv")
    ap.add_argument("--full_grid", action="store_true",
                    help="digital scheme on every (C, SNR) pair and both codec banks (slow, ~24 h); "
                         "default = only the points shown in the figures")


def main():
    args, cfg, dev = setup(__doc__, extra)
    w, tr, p = cfg["witt"], cfg["train"], cfg["vit"]["patch"]
    h0, w0 = cfg["data"]["native_hw"]
    clf = load_classifier(cfg, dev)
    test_dl = get_loader(cfg, "test", batch_size=cfg["eval"]["batch_size"], shuffle=False)
    rows = []
    native = w["input"] == "native"
    hw0 = tuple(cfg["data"]["native_hw"] if native else cfg["data"]["input_hw"])
    rhos = {C: rt.bandwidth_ratio(channel_uses(C, hw0, 2 if native else 4), (h0, w0)) for C in w["C_list"]}
    for C in ([] if args.digital_only else w["C_list"]):
        net, hw = build_witt(cfg, C, dev)
        n_cu = channel_uses(C, hw, 2 if native else 4)
        print(f"WITT C={C}: {n_cu:.0f} channel uses/image, rho(native)={rhos[C]:.4f}")
        ck = out_path(cfg, "ckpt", f"witt_{w['input']}_C{C}.pt")
        if not args.skip_train and not os.path.exists(ck):
            opt = torch.optim.Adam(net.parameters(), lr=w["lr"])
            train_dl = get_loader(cfg, "train")
            n_bad = 0
            for ep in range(tr["witt_epochs"]):
                net.train()
                for x, _ in limited(train_dl, tr["max_steps_per_epoch"]):
                    x = x.to(dev) if native else to_input(x.to(dev), cfg)
                    _, _, _, _, loss = net(x)
                    if not torch.isfinite(loss):  # guard against divergence
                        n_bad += 1
                        opt.zero_grad(set_to_none=True)
                        continue
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                print(f"  epoch {ep}: loss {loss.item():.5f}  skipped non-finite steps so far: {n_bad}", flush=True)
            save_ckpt(net, ck)
        elif os.path.exists(ck):
            load_ckpt(net, ck)
        net.eval()
        for snr in cfg["channel"]["eval_snrs_db"]:
            torch.manual_seed(snr)
            for x, y in limited(test_dl, tr["max_steps_per_epoch"]):
                with torch.no_grad():
                    x_nat = x.to(dev)
                    xin = x_nat if native else to_input(x_nat, cfg)
                    rec, _, _, _, _ = net(xin, given_SNR=snr)
                    rec = rec.clamp(0, 1)
                    rec_in = to_input(rec, cfg) if native else rec
                    pred = clf(rec_in).argmax(1).cpu()
                    rec_nat = rec if native else resize(rec, (h0, w0), "area")
                    ps = psnr(x_nat, rec_nat).cpu()
                for i in range(len(y)):
                    rows.append(dict(method="WITT", C=C, rho=rhos[C], snr_db=snr, n_cu=n_cu,
                                     correct=int(pred[i] == y[i]), psnr_native=float(ps[i])))
        wdf = pd.DataFrame([r for r in rows if r["C"] == C])
        print(f"WITT C={C} summary: acc@max SNR {wdf[wdf.snr_db == wdf.snr_db.max()].correct.mean():.4f}, "
              f"PSNR {wdf.psnr_native.mean():.2f} dB, non-finite PSNR {int((~np.isfinite(wdf.psnr_native)).sum())}", flush=True)
    wpath = out_path(cfg, "results", "s7_witt_per_image.csv")
    if not args.digital_only:
        pd.DataFrame(rows).to_csv(wpath, index=False)

    if args.skip_digital:
        return
    # ---------------- proposed digital scheme at matched channel uses
    m = cfg["mr"]
    R = list(m["levels_bytes"])
    vit = load_vit(cfg, dev)
    banks = {"MR-native (proposed)": load_codec_bank(cfg, dev, domain="native")}
    if args.full_grid:  # the upsampled bank cannot reach these budgets (side info alone is 902 B, lowest level ~30 kB)
        banks["MR (proposed)"] = load_codec_bank(cfg, dev, domain="upsampled")
    snrs = list(cfg["channel"]["eval_snrs_db"])
    snr0 = min(snrs, key=lambda v: abs(v - cfg["channel"]["awgn_snr_db"]))  # same choice as make_figures.py
    C_mid = min(rhos, key=lambda c: abs(rhos[c] - 0.25))
    drows = []
    configs = []
    for C, rho in rhos.items():
        n_cu = rho * 3 * h0 * w0
        for snr in snrs:
            if not args.full_grid and snr != snr0 and C != C_mid:
                continue  # figure uses: every rho at snr0, and every SNR at rho=0.25
            for eta_name, eta in (("capacity", float(ch.capacity_eta(snr))),
                                  ("MCS", float(ch.mcs_eta(snr, cfg["channel"]["mcs_margin_db"])))):
                configs.append((C, rho, snr, eta_name, n_cu * eta / 8))
    print(f"digital scheme: {len(configs)} (C, SNR, eta) points x {len(banks)} bank(s)")
    from tqdm import tqdm
    for x, y in tqdm(limited(test_dl, tr["max_steps_per_epoch"]), total=len(test_dl), desc="MR at matched channel uses"):
        with torch.no_grad():
            x_nat = x.to(dev)
            x_in = to_input(x_nat, cfg)
            _, s = importance(vit, x_in, p)  # once per batch, reused for every configuration
            for C, rho, snr, eta_name, budget in configs:
                for name, bank in banks.items():
                    xc = x_in if "native" not in name else x_nat
                    rec, _, bits, _ = mr_transmit(bank, xc, s, budget, R, m["header_bits"], "feasible", p)
                    rec_in = rec if "native" not in name else to_input(rec, cfg)
                    pred = clf(rec_in).argmax(1).cpu()
                    rec_nat = resize(rec, (h0, w0), "area") if "native" not in name else rec
                    ps = psnr(x_nat, rec_nat).cpu()
                    for i in range(len(y)):
                        drows.append(dict(method=f"{name}, {eta_name}", C=C, rho=rho, snr_db=snr,
                                          budget_bytes=budget, bits=float(bits[i]),
                                          correct=int(pred[i] == y[i]), psnr_native=float(ps[i])))
    pd.DataFrame(drows).to_csv(out_path(cfg, "results", "s7_digital_matched_per_image.csv"), index=False)


if __name__ == "__main__":
    main()

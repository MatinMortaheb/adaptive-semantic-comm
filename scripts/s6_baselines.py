"""S6 - External baselines on a common, bit-accurate axis.

Methods (all rates include side information; bpp per native 64x64 pixel):
  MR (proposed, upsampled codecs)       - paper's scheme with corrected Algorithm 2
  MR-native (proposed, native codecs)   - variant: importance from the ViT on 320x480,
                                          attention pooled to the native 8x8 patch grid, patches coded at 64x64
  Uniform / Uniform-native              - same codec bank, no importance
  IAQ / IAQ-native                      - Park et al. IoT-J 2025, same attention map
  JPEG / JPEG2000 / WebP / BPG          - on the 320x480 input
  Native-* (raw, JPEG, JPEG2000, WebP, BPG) - native 64x64 image coded, bicubic-upsampled at the receiver
The binary scheme (S1) and WITT (S7) are added to the same figure by make_figures.py."""
from _common import setup

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from semcomm import rate as rt
from semcomm import std_codecs
from semcomm.data import get_loader, resize, to_input
from semcomm.metrics import psnr
from semcomm.system import (iaq_transmit, importance, load_classifier, load_codec_bank, load_vit, mr_transmit,
                            stdcodec_transmit, uniform_transmit)
from semcomm.utils import limited, out_path


def extra(ap):
    ap.add_argument("--skip", nargs="*", default=[], help="method families to skip: mr uniform iaq std native")


def main():
    args, cfg, dev = setup(__doc__, extra)
    m, bl, p = cfg["mr"], cfg["baselines"], cfg["vit"]["patch"]
    R = list(m["levels_bytes"])
    H, W = cfg["data"]["input_hw"]
    h0, w0 = cfg["data"]["native_hw"]
    P_up, P_nat = (H // p) * (W // p), (h0 // p) * (w0 // p)
    vit, clf = load_vit(cfg, dev), load_classifier(cfg, dev)
    bank_up = load_codec_bank(cfg, dev, domain="upsampled")
    bank_nat = load_codec_bank(cfg, dev, domain="native")
    codecs = std_codecs.available_codecs()
    print("standard codecs available:", codecs)
    qgrid = {"jpeg": bl["jpeg_qualities"], "jp2": bl["jp2_rates"], "webp": bl["webp_qualities"], "bpg": bl["bpg_qps"]}
    dl = get_loader(cfg, "test", batch_size=cfg["eval"]["batch_size"], shuffle=False)
    rows = []
    rng = np.random.default_rng(0)

    def record(method, param, x_nat, x_in, rec_in, bits, y):
        with torch.no_grad():
            pred = clf(rec_in).argmax(1).cpu()
            ps_in = psnr(x_in, rec_in).cpu()
            rec_nat = resize(rec_in, (h0, w0), "area")
            ps_nat = psnr(x_nat, rec_nat).cpu()
        for i in range(len(y)):
            rows.append(dict(method=method, param=param, bits=float(bits[i]),
                             bpp_native=rt.bpp(bits[i], (h0, w0)), correct=int(pred[i] == y[i]),
                             psnr_input=float(ps_in[i]), psnr_native=float(ps_nat[i])))

    for x, y in limited(dl, cfg["train"]["max_steps_per_epoch"]):
        x_nat = x.to(dev)
        with torch.no_grad():
            x_in = to_input(x_nat, cfg)
            _, s = importance(vit, x_in, p)
            up = lambda z: to_input(z, cfg)  # noqa: E731
            for f in m["rate_fracs"]:
                b_up, b_nat = f * P_up * R[-1], f * P_nat * R[-1]
                if "mr" not in args.skip:
                    rec, _, bits, _ = mr_transmit(bank_up, x_in, s, b_up, R, m["header_bits"], "feasible", p)
                    record("MR (proposed)", f, x_nat, x_in, rec, bits, y)
                    rec, _, bits, _ = mr_transmit(bank_nat, x_nat, s, b_nat, R, m["header_bits"], "feasible", p)
                    record("MR-native (proposed)", f, x_nat, x_in, up(rec), bits, y)
                if "uniform" not in args.skip:
                    rec, _, bits = uniform_transmit(bank_up, x_in, b_up, R, rng, p)
                    record("Uniform", f, x_nat, x_in, rec, bits, y)
                    rec, _, bits = uniform_transmit(bank_nat, x_nat, b_nat, R, rng, p)
                    record("Uniform-native", f, x_nat, x_in, up(rec), bits, y)
                if "iaq" not in args.skip:
                    rec, bits = iaq_transmit(x_in, s, b_up, p, bl["iaq_max_bits"])
                    record("IAQ [Park]", f, x_nat, x_in, rec, bits, y)
                    rec, bits = iaq_transmit(x_nat, s, b_nat, p, bl["iaq_max_bits"])
                    record("IAQ-native [Park]", f, x_nat, x_in, up(rec), bits, y)
        if "std" not in args.skip:
            for c in [c for c in codecs if c != "raw"]:
                for q in qgrid[c]:
                    rec, bits = stdcodec_transmit(x_in, c, q)
                    record(c.upper(), q, x_nat, x_in, rec.to(dev), bits, y)
        if "native" not in args.skip:
            for c in codecs:
                for q in (qgrid.get(c) or [None]):
                    rec, bits = stdcodec_transmit(x_nat, c, q)
                    record(f"Native-{c.upper()}", q, x_nat, x_in, up(rec.to(dev)), bits, y)
    df = pd.DataFrame(rows)
    df.to_csv(out_path(cfg, "results", "s6_baselines_per_image.csv"), index=False)
    summ = df.groupby(["method", "param"]).mean(numeric_only=True).reset_index()
    summ.to_csv(out_path(cfg, "results", "s6_baselines_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.sort_values(["method", "bpp_native"]).to_string())


if __name__ == "__main__":
    main()

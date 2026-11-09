"""S3 - Multi-resolution scheme with the corrected Algorithm 2.

(a) single fixed resolution, top-q patches by attention   (upsampled codecs)
(b) adaptive multi-resolution vs budget                    (upsampled codecs)
(c) legacy vs corrected Algorithm 2: budget overshoot (Table 'alg2_overshoot')
All rates include side information; bpp is per native pixel."""
from _common import setup

import numpy as np
import pandas as pd
import torch

from semcomm import allocation as alloc
from semcomm import rate as rt
from semcomm.data import get_loader, patchify, to_input, unpatchify
from semcomm.metrics import psnr
from semcomm.system import importance, load_classifier, load_codec_bank, load_vit, mr_transmit
from semcomm.utils import limited, out_path


def main():
    args, cfg, dev = setup(__doc__)
    m, p = cfg["mr"], cfg["vit"]["patch"]
    R = list(m["levels_bytes"])
    H, W = cfg["data"]["input_hw"]
    P = (H // p) * (W // p)
    R_full = P * R[-1]
    vit, clf = load_vit(cfg, dev), load_classifier(cfg, dev)
    bank = load_codec_bank(cfg, dev, domain="upsampled")
    dl = get_loader(cfg, "test", batch_size=cfg["eval"]["batch_size"], shuffle=False)
    nat = cfg["data"]["native_hw"]
    fixed, adapt, over, hist = [], [], [], []
    for x, y in limited(dl, cfg["train"]["max_steps_per_epoch"]):
        with torch.no_grad():
            x_in = to_input(x.to(dev), cfg)
            _, s = importance(vit, x_in, p)
            # (a) fixed single resolution on top-q patches
            order = s.flatten(1).argsort(1, descending=True)
            for b in R[1:-1]:
                for q in m["fixed_res_q"]:
                    n = int(round(q * P))
                    sizes = torch.zeros(x.shape[0], P, dtype=torch.long, device=dev)
                    sizes.scatter_(1, order[:, :n], b)
                    rec, pbits = bank.reconstruct(patchify(x_in, p), sizes, return_bits=True)
                    rec = unpatchify(rec, H, W)
                    pred = clf(rec).argmax(1).cpu()
                    bits = pbits.sum(1).cpu().numpy() + rt.binary_mask_bits(P, n) + m["header_bits"]
                    for i in range(len(y)):
                        fixed.append(dict(level=b, q=q, correct=int(pred[i] == y[i]),
                                          psnr_full=float(psnr(x_in[i:i+1], rec[i:i+1])), bits=float(bits[i]),
                                          bpp_native=rt.bpp(bits[i], nat)))
            # (b) adaptive + (c) overshoot of the legacy rule
            for f in m["rate_fracs"]:
                budget = f * R_full
                rec, lv, bits, _ = mr_transmit(bank, x_in, s, budget, R, m["header_bits"], "feasible", p)
                pred = clf(rec).argmax(1).cpu()
                ps = psnr(x_in, rec).cpu()
                ent_bits = [rt.mr_rate_bits(l, R, m["header_bits"], entropy_coded=True)[0] for l in lv]
                for i in range(len(y)):
                    adapt.append(dict(rate_frac=f, budget_bytes=budget, correct=int(pred[i] == y[i]),
                                      psnr_full=float(ps[i]), bits=int(bits[i]), bits_entropy_side=int(ent_bits[i]),
                                      bpp_native=rt.bpp(bits[i], nat), bpp_native_entropy=rt.bpp(ent_bits[i], nat)))
                    hist.append(dict(rate_frac=f, **{f"L{k}": float((lv[i] == k).mean()) for k in range(len(R))}))
                # legacy rule (allocation only, no decoding needed); it ignored side info, so report the
                # overshoot of the PAYLOAD over the budget it was given
                lv_old = np.stack([alloc.alg2_legacy(s[i].flatten().double().cpu().numpy(), budget, R) for i in range(len(y))])
                pay_old = np.asarray(R)[lv_old].sum(1)
                for i in range(len(y)):
                    over.append(dict(rate_frac=f, budget_bytes=budget, legacy_payload=float(pay_old[i]),
                                     legacy_over=float(pay_old[i] - budget),
                                     legacy_over_with_side=float(pay_old[i] + rt.mr_side_bits_fixed(P, len(R), m["header_bits"]) / 8 - budget)))
    res = out_path(cfg, "results")
    pd.DataFrame(fixed).to_csv(f"{res}/s3_fixed_res.csv", index=False)
    pd.DataFrame(adapt).to_csv(f"{res}/s3_adaptive.csv", index=False)
    pd.DataFrame(hist).groupby("rate_frac").mean().reset_index().to_csv(f"{res}/s3_level_hist.csv", index=False)
    ov = pd.DataFrame(over)
    ov.to_csv(f"{res}/s3_alg2_overshoot_per_image.csv", index=False)
    g = ov.groupby("rate_frac")
    pd.DataFrame({"frac_images_over_budget": g.apply(lambda d: (d.legacy_over > 1e-6).mean()),
                  "mean_over_bytes": g.apply(lambda d: d.legacy_over.clip(lower=0).mean()),
                  "max_over_bytes": g.legacy_over.max(),
                  "frac_over_incl_side": g.apply(lambda d: (d.legacy_over_with_side > 1e-6).mean())}
                 ).reset_index().to_csv(f"{res}/s3_alg2_overshoot.csv", index=False)
    print(pd.DataFrame(adapt).groupby("rate_frac").mean(numeric_only=True)[["correct", "psnr_full", "bpp_native"]])


if __name__ == "__main__":
    main()

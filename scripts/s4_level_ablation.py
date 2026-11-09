"""S4 - Ablation on the number and values of resolution levels + codec overhead (codecs on the upsampled image)."""
from _common import setup

import os

import pandas as pd
import torch

from semcomm import rate as rt
from semcomm.data import get_loader, to_input
from semcomm.system import importance, load_classifier, load_codec_bank, load_vit, mr_transmit
from semcomm.utils import limited, load_json, out_path


def main():
    args, cfg, dev = setup(__doc__)
    m, p = cfg["mr"], cfg["vit"]["patch"]
    H, W = cfg["data"]["input_hw"]
    P = (H // p) * (W // p)
    raw = 3 * p * p
    vit, clf = load_vit(cfg, dev), load_classifier(cfg, dev)
    dl = get_loader(cfg, "test", batch_size=cfg["eval"]["batch_size"], shuffle=False)
    s2 = out_path(cfg, "results", "s2_codecs_upsampled.json")
    cod = load_json(s2) if os.path.exists(s2) else {}
    rows = []
    for R in m["ablation_sets"]:
        bank = load_codec_bank(cfg, dev, sizes=R, domain="upsampled")
        corr = {f: [] for f in m["eval_rate_fracs_ablation"]}
        bpps = {f: [] for f in m["eval_rate_fracs_ablation"]}
        for x, y in limited(dl, cfg["train"]["max_steps_per_epoch"]):
            with torch.no_grad():
                x_in = to_input(x.to(dev), cfg)
                _, s = importance(vit, x_in, p)
                for f in corr:
                    rec, lv, bits, _ = mr_transmit(bank, x_in, s, f * P * raw, R, m["header_bits"], "feasible", p)
                    corr[f] += (clf(rec).argmax(1).cpu() == y).tolist()
                    bpps[f] += [rt.bpp(b_, cfg["data"]["native_hw"]) for b_ in bits]
        inter = [b for b in R if b not in (0, raw)]
        row = {"levels": "{" + ",".join(map(str, R)) + "}", "n_levels": len(R)}
        for f in corr:
            row[f"acc@{f}"] = sum(corr[f]) / max(len(corr[f]), 1)
            row[f"bpp@{f}"] = sum(bpps[f]) / max(len(bpps[f]), 1)
        row["codec_params_K"] = sum(cod.get(str(b), {}).get("params", float("nan")) for b in inter) / 1e3
        row["enc_params_K"] = sum(cod.get(str(b), {}).get("enc_params", float("nan")) for b in inter) / 1e3
        row["train_gpu_hours"] = sum(cod.get(str(b), {}).get("train_seconds", float("nan")) for b in inter) / 3600
        row["storage_MB_fp32"] = row["codec_params_K"] * 1e3 * 4 / 2 ** 20
        rows.append(row)
        print(row)
    pd.DataFrame(rows).to_csv(out_path(cfg, "results", "s4_level_ablation.csv"), index=False)


if __name__ == "__main__":
    main()

"""S5 - Bit-level rate accounting table. Analytic; uses S3 results
(if present) for the entropy-coded side-information column. Runs without a GPU or dataset."""
from _common import setup

import os

import pandas as pd

from semcomm import rate as rt
from semcomm.utils import out_path


def main():
    args, cfg, dev = setup(__doc__)
    p = cfg["vit"]["patch"]
    H, W = cfg["data"]["input_hw"]
    P = (H // p) * (W // p)
    R = list(cfg["mr"]["levels_bytes"])
    D, bz, hdr = 384, cfg["binary"]["token_bits"], cfg["mr"]["header_bits"]
    nat = cfg["data"]["native_hw"]
    Rfull = P * R[-1]
    rows = []
    for r in cfg["binary"]["rates"]:
        n = int(round(r * P))
        pay = n * D * bz / 8
        side = (rt.binary_mask_bits(P, n) + 64) / 8  # mask + 2x32-bit token range
        rows.append(("Binary", f"r={r}", pay, side, None))
    side_fixed = rt.mr_side_bits_fixed(P, len(R), hdr) / 8
    ent = {}
    s3 = out_path(cfg, "results", "s3_adaptive.csv")
    if os.path.exists(s3):
        d = pd.read_csv(s3)
        d["side_ent"] = (d.bits_entropy_side - (d.bits - rt.mr_side_bits_fixed(P, len(R), hdr))) / 8
        ent = d.groupby("rate_frac").side_ent.mean().to_dict()
    for f in [R[1] * P / Rfull + 1e-9, 0.125, 0.25, 0.5, 1.0]:
        budget = f * Rfull
        pay = budget - side_fixed if f < 1 else Rfull
        label = "all patches at $R_1$" if abs(f - R[1] / R[-1]) < 1e-6 else f"budget={f:g} R_full"
        if abs(f - R[1] / R[-1]) < 1e-6:
            pay = R[1] * P
        rows.append(("MR", label, pay, side_fixed, ent.get(round(f, 6))))
    out = []
    for scheme, op, pay, side, side_e in rows:
        tot = pay + side
        out.append(dict(scheme=scheme, operating_point=op, payload_bytes=round(pay), side_bytes=round(side),
                        side_bytes_entropy=None if side_e is None else round(side_e),
                        total_bytes=round(tot), side_pct=100 * side / tot,
                        bpp_input=8 * tot / (H * W), bpp_native=rt.bpp(8 * tot, nat),
                        raw_native_bytes=3 * nat[0] * nat[1], ratio_to_raw_native=tot / (3 * nat[0] * nat[1])))
    df = pd.DataFrame(out)
    df.to_csv(out_path(cfg, "results", "s5_rate_table.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df)


if __name__ == "__main__":
    main()

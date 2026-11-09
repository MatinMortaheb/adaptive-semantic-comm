"""End-to-end transmit/receive for the proposed schemes and the baselines.

Conventions
  x_nat : (B,3,64,64) native TinyImageNet image in [0,1]
  x_in  : (B,3,320,480) network input (paper's resizing)
  scores: (B, Gh, Gw) importance map (head-averaged CLS attention) on the 40x60 grid
Every function returns the reconstruction at 320x480 (what the classifier sees) and the
exact number of transmitted bits (payload + side information).
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import allocation as alloc
from . import rate as rt
from . import std_codecs
from .data import patchify, resize, unpatchify
from .models import cls_attention_map


# ------------------------------------------------------------------ loading
def load_vit(cfg, device):
    from .models import ViTWithHead
    from .utils import load_ckpt, out_path
    import os
    vit = ViTWithHead(cfg)
    ck = cfg["vit"].get("ckpt") or out_path(cfg, "ckpt", "vit_finetuned.pt")
    if os.path.exists(ck):
        load_ckpt(vit, ck)
    else:
        print(f"[warn] no fine-tuned ViT at {ck}; using DINO init + random head (run scripts/s0_finetune_vit.py)")
    return vit.to(device).eval()


def load_classifier(cfg, device):
    from .models import make_classifier
    from .utils import load_ckpt, out_path
    import os
    clf = make_classifier(cfg)
    if cfg["classifier"]["arch"] == "hf_vit":  # pretrained Hugging Face model, nothing to load
        return clf.to(device).eval()
    ck = cfg["classifier"].get("ckpt") or out_path(cfg, "ckpt", "classifier.pt")
    if os.path.exists(ck):
        load_ckpt(clf, ck)
    else:
        print(f"[warn] no classifier at {ck} (run scripts/s0b_train_classifier.py)")
    return clf.to(device).eval()


def load_codec_bank(cfg, device, sizes=None, domain=None):
    from .models import CodecBank
    from .utils import load_ckpt, out_path
    import os
    m = cfg["mr"]
    domain = domain or m["codec_domain"]
    sizes = sizes or m["levels_bytes"]
    bank = CodecBank(sizes, p=cfg["vit"]["patch"], bits=m["latent_bits"], act=m["latent_activation"])
    for b, c in bank.codecs.items():
        ck = out_path(cfg, "ckpt", f"codec_{domain}_{b}.pt")
        if os.path.exists(ck):
            load_ckpt(c, ck)
        else:
            print(f"[warn] missing codec {ck} (run scripts/s2_train_codecs.py)")
    bank = bank.to(device).eval()
    ep = out_path(cfg, "ckpt", f"entropy_{domain}.npz")
    if cfg.get("rate", {}).get("entropy_coding", True) and os.path.exists(ep):
        from .entropy import EntropyModel
        bank.ent = EntropyModel.load(ep, device)
        missing = [b for b in sizes if b not in (0,) and int(b) not in bank.ent.nlog2p]
        if missing:
            print(f"[warn] entropy model {ep} has no pmf for sizes {missing}: fixed-length for those")
    elif cfg.get("rate", {}).get("entropy_coding", True):
        print(f"[warn] no entropy model at {ep} (run scripts/s2b_fit_entropy.py); using fixed-length rates")
    return bank


@torch.no_grad()
def importance(vit, x_in, p=8):
    z, attn = vit.tokens_and_attention(x_in)
    Gh, Gw = x_in.shape[-2] // p, x_in.shape[-1] // p
    return z, cls_attention_map(attn, (Gh, Gw))


# ------------------------------------------------------------ binary scheme
def binary_transmit(decoder, z, scores, r, alpha, rng, D=384, token_bits=8, grid=(40, 60)):
    """z: (B,P,D) tokens; scores (B,Gh,Gw). Returns x_hat (B,3,320,480), sel (B,P) bool, bits (B,)."""
    B, P, _ = z.shape
    n = int(round(r * P))
    sel = np.stack([alloc.binary_select(scores[b].flatten().cpu().numpy(), n, alpha, rng) for b in range(B)])
    m = torch.from_numpy(sel).to(z.device).unsqueeze(-1).float()
    zq = z
    if token_bits:
        # per-image min/max uniform quantization of transmitted tokens (b_z bits/element; 2x32-bit range sent)
        lo = z.amin(dim=(1, 2), keepdim=True)
        hi = z.amax(dim=(1, 2), keepdim=True)
        L = 2 ** token_bits - 1
        zq = torch.round((z - lo) / (hi - lo + 1e-8) * L) / L * (hi - lo) + lo
    zg = (zq * m).transpose(1, 2).reshape(B, D, *grid)
    x_hat = decoder(zg)
    bits = np.array([rt.binary_rate_bits(P, int(s.sum()), D, token_bits) + 64 for s in sel])
    return x_hat, sel, bits


def mask_to_pixels(sel, grid, p):
    """(B,P) bool -> (B,1,H,W) float."""
    B = sel.shape[0]
    m = torch.as_tensor(sel, dtype=torch.float32).view(B, 1, *grid)
    return F.interpolate(m, scale_factor=p, mode="nearest")


# ------------------------------------------------------ multi-resolution
def mr_transmit(bank, x_code, scores, budget_bytes, R, header_bits=16, algorithm="feasible",
                p=8, side_entropy=None):
    """x_code : image on which the patch codecs operate (320x480 'upsampled' or 64x64 'native')
    scores : (B,Gh,Gw) importance; pooled to x_code's patch grid if needed.
    If the bank has an entropy model (bank.ent), Algorithm 2 allocates with the expected
    entropy-coded cost of each level and the reported rate is the entropy-coded payload plus the
    entropy-coded resolution map; otherwise fixed-length sizes are used throughout.
    Returns x_rec (same size as x_code), levels (B,P), bits (B,), overshoot_bytes (B,)."""
    B, _, H, W = x_code.shape
    gh, gw = H // p, W // p
    if scores.shape[-2:] != (gh, gw):
        scores = F.adaptive_avg_pool2d(scores.unsqueeze(1), (gh, gw)).squeeze(1)
    P = gh * gw
    L = len(R)
    ent = bank.ent is not None
    if side_entropy is None:
        side_entropy = ent
    costs = bank.level_costs(R)
    side = rt.mr_side_bits_fixed(P, L, header_bits) / 8.0  # conservative (fixed-length) in the budget
    levels, over = [], []
    for b in range(B):
        s = scores[b].flatten().double().cpu().numpy()
        if algorithm == "feasible":
            lv = alloc.alg2_feasible(s, budget_bytes, costs, side_bytes=side)
        elif algorithm == "legacy":
            lv = alloc.alg2_legacy(s, budget_bytes, R)
        else:
            raise ValueError(algorithm)
        levels.append(lv)
        over.append(np.asarray(R)[lv].sum() + side - budget_bytes)
    levels = np.stack(levels)
    sizes = torch.as_tensor(np.asarray(R)[levels], device=x_code.device)
    rec, pbits = bank.reconstruct(patchify(x_code, p), sizes, return_bits=True)
    rec = unpatchify(rec, H, W)
    payload = pbits.sum(1).cpu().numpy()
    sidebits = np.array([rt.mr_side_bits_entropy(lv, L, header_bits) if side_entropy
                         else rt.mr_side_bits_fixed(P, L, header_bits) for lv in levels])
    bits = payload + sidebits
    if budget_bytes < side:  # cannot even afford the side information: nothing is sent
        bits[:] = 0
    return rec, levels, bits, np.array(over)


def uniform_transmit(bank, x_code, budget_bytes, R, rng, p=8):
    """Importance-agnostic baseline with the same codec bank (and the same entropy model if attached)."""
    B, _, H, W = x_code.shape
    P = (H // p) * (W // p)
    costs = bank.level_costs(R)
    lv = np.stack([alloc.uniform_levels(P, budget_bytes - 2, costs, rng) for _ in range(B)])  # 2-byte header
    sizes = torch.as_tensor(np.asarray(R)[lv], device=x_code.device)
    rec, pbits = bank.reconstruct(patchify(x_code, p), sizes, return_bits=True)
    rec = unpatchify(rec, H, W)
    bits = pbits.sum(1).cpu().numpy() + 16
    return rec, lv, bits


def iaq_transmit(x_code, scores, budget_bytes, p=8, max_bits=8):
    """Park et al. IAQ: per-patch uniform pixel quantization, bits allocated by attention."""
    B, _, H, W = x_code.shape
    gh, gw = H // p, W // p
    if scores.shape[-2:] != (gh, gw):
        scores = F.adaptive_avg_pool2d(scores.unsqueeze(1), (gh, gw)).squeeze(1)
    P = gh * gw
    side_bits = P * 4 + 16  # M_i in {0..8}: 4 bits each + header
    patches = patchify(x_code, p)
    out = torch.empty_like(patches)
    bits = []
    for b in range(B):
        pt = patches[b].reshape(P, -1).cpu().numpy()
        M = alloc.iaq_allocate(pt, scores[b].flatten().double().cpu().numpy(),
                               max(0, budget_bytes * 8 - side_bits), max_bits)
        Lv = (2.0 ** M)[:, None]  # vectorized uniform mid-rise quantization, M_i bits per pixel (M=0 -> 0.5)
        q = np.where(M[:, None] == 0, 0.5, (np.floor(np.clip(pt, 0, 1 - 1e-9) * Lv) + 0.5) / Lv)
        out[b] = torch.from_numpy(q).view_as(patches[b]).to(out)
        bits.append(pt.shape[1] * M.sum() + side_bits)
    return unpatchify(out, H, W), np.array(bits)


def stdcodec_transmit(x, codec, q):
    """x: (B,3,h,w) -> reconstructions, bits."""
    recs, bits = [], []
    for b in range(x.shape[0]):
        r, n = std_codecs.encode_decode(x[b].cpu(), codec, q)
        recs.append(r)
        bits.append(8 * n)
    return torch.stack(recs).to(x.device), np.array(bits)

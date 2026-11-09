"""Networks used in the paper.

* ViTWithHead      : DINO ViT-S/8 (init) + linear head, fine-tuned with CE (Algorithm 1, encoder training)
* BinaryDecoder    : convolutional decoder of Fig. 22(a)
* PatchEncoder/Decoder : per-resolution codec of Fig. 22(b)
* make_classifier  : downstream classifier applied to reconstructed images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dino_vit


# --------------------------------------------------------------------------- ViT
class ViTWithHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v = cfg["vit"]
        self.backbone = dino_vit.__dict__[v["arch"]](patch_size=v["patch"], num_classes=0)
        if v.get("pretrained") and not v.get("ckpt"):
            sd = torch.hub.load_state_dict_from_url(v["weights_url"], map_location="cpu")
            self.backbone.load_state_dict(sd, strict=True)
        self.head = nn.Linear(self.backbone.embed_dim, cfg["data"]["num_classes"])
        self.p = v["patch"]
        # DINO was trained on ImageNet-normalized inputs; images here are in [0,1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.head(self.backbone((x - self.mean) / self.std))

    @torch.no_grad()
    def tokens_and_attention(self, x):
        """Returns
        z    : (B, P, D) last-layer patch outputs (after final norm) -> transmitted tokens (binary scheme)
        attn : (B, H, P+1, P+1) last-layer attention -> CLS row gives patch importance
        """
        bb = self.backbone
        t = bb.prepare_tokens((x - self.mean) / self.std)  # interpolates positional encoding to the 40x60 grid
        for blk in bb.blocks[:-1]:
            t = blk(t)
        last = bb.blocks[-1]
        y, attn = last.attn(last.norm1(t))
        t = t + last.drop_path(y)
        t = t + last.drop_path(last.mlp(last.norm2(t)))
        t = bb.norm(t)
        return t[:, 1:], attn


def cls_attention_map(attn, grid_hw):
    """Average over heads of A^{(h)}[0, 1:], reshaped to the patch grid (Fig. 15/16, 19)."""
    a = attn[:, :, 0, 1:].mean(1)  # (B, P)
    return a.reshape(a.shape[0], *grid_hw)


# ---------------------------------------------------------------- binary decoder
class BinaryDecoder(nn.Module):
    """Fig. 22(a): Conv(384,256)-BN-ReLU-...-Conv(8,3).

    ASSUMPTION: Fig. 22(a) lists only stride-1 3x3 convolutions, which cannot map the
    40x60 token grid to a 320x480 image. We insert x2 nearest upsampling before the
    last three stages (40x60 -> 320x480). Replace with your original decoder if it differs
    (see README, 'Plugging in your own modules')."""

    def __init__(self, D=384, up_stages=3):
        super().__init__()
        ch = [D, 256, 128, 64, 32, 16, 8]
        layers = []
        n = len(ch) - 1
        for i in range(n):
            if i >= n - up_stages:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(True)]
        layers += [nn.Conv2d(8, 3, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z_grid):  # (B, D, 40, 60)
        return self.net(z_grid)


# ------------------------------------------------------------- quantizer
class UniformQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        L = 2 ** bits - 1
        return torch.round(x.clamp(0, 1) * L) / L

    @staticmethod
    def backward(ctx, g):
        return g, None


def quantize(x, bits):
    return UniformQuantSTE.apply(x, bits)


# ------------------------------------------------------------ per-patch codecs
class PatchEncoder(nn.Module):
    """Fig. 22(b) encoder. Output: b values, bounded to [0,1] and quantized to `bits` each."""

    def __init__(self, b, p=8, bits=8, act="sigmoid"):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Flatten(), nn.Linear(256 * p * p, 512), nn.Linear(512, b))
        self.bits, self.act = bits, act

    def forward(self, x):
        y = self.f(x)
        y = torch.sigmoid(y) if self.act == "sigmoid" else y.clamp(0, 1)
        return quantize(y, self.bits)


class PatchDecoder(nn.Module):
    """Fig. 22(b) decoder."""

    def __init__(self, b, p=8):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(b, 512), nn.Linear(512, 256 * p * p), nn.Unflatten(1, (256, p, p)),
            nn.ConvTranspose2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, padding=1))

    def forward(self, w):
        return self.f(w)


class PatchCodec(nn.Module):
    def __init__(self, b, p=8, bits=8, act="sigmoid"):
        super().__init__()
        self.b = b
        self.enc = PatchEncoder(b, p, bits, act)
        self.dec = PatchDecoder(b, p)

    def forward(self, x):
        return self.dec(self.enc(x))


class CodecBank(nn.Module):
    """Maps a level size in bytes to a codec. 0 -> zeros, 3*p*p -> raw 8-bit patch."""

    def __init__(self, sizes, p=8, bits=8, act="sigmoid"):
        super().__init__()
        self.p = p
        self.raw = 3 * p * p
        self.codecs = nn.ModuleDict({str(b): PatchCodec(b, p, bits, act)
                                     for b in sizes if b not in (0, self.raw)})

    ent = None  # optional EntropyModel (semcomm/entropy.py), attached by system.load_codec_bank

    def symbols(self, patches, b):
        """8-bit integer symbols that are actually transmitted for level b: (N, dims)."""
        if b == self.raw:
            return torch.round(patches.clamp(0, 1) * 255).flatten(1).long()
        return torch.round(self.codecs[str(b)].enc(patches) * 255).long()

    def code(self, patches, b):
        """patches: (N,3,p,p) -> reconstructions with level b bytes."""
        if b == 0:
            return torch.zeros_like(patches)
        if b == self.raw:
            return torch.round(patches.clamp(0, 1) * 255) / 255
        return self.codecs[str(b)](patches).clamp(0, 1)

    def reconstruct(self, patches, size_per_patch, return_bits=False):
        """patches: (B,P,3,p,p); size_per_patch: (B,P) long tensor of byte sizes.
        return_bits: also return (B,P) payload bits per patch -- entropy-coded if an entropy
        model is attached, otherwise fixed-length (8 * size)."""
        out = torch.zeros_like(patches)
        flat, sz = patches.flatten(0, 1), size_per_patch.flatten()
        o = out.flatten(0, 1)
        bits = torch.zeros(sz.shape, dtype=torch.float32, device=patches.device)
        for b in sz.unique().tolist():
            idx = (sz == b).nonzero(as_tuple=True)[0]
            if b != 0 and len(idx):
                if b == self.raw:
                    o[idx] = torch.round(flat[idx].clamp(0, 1) * 255) / 255
                    w = None
                else:
                    w = self.codecs[str(b)].enc(flat[idx])
                    o[idx] = self.codecs[str(b)].dec(w).clamp(0, 1)
                if return_bits:
                    if self.ent is not None and int(b) in self.ent.nlog2p:
                        sym = (torch.round(w * 255).long() if w is not None
                               else torch.round(flat[idx].clamp(0, 1) * 255).flatten(1).long())
                        bits[idx] = self.ent.bits(b, sym)
                    else:
                        bits[idx] = 8.0 * b
        rec = o.view_as(patches)
        return (rec, bits.view(size_per_patch.shape)) if return_bits else rec

    def level_costs(self, R):
        """bytes per patch used by the allocator for each level: expected entropy-coded size if an
        entropy model is attached (so the budget is filled correctly), otherwise the nominal size."""
        if self.ent is None:
            return list(R)
        c = [0.0 if b == 0 else (self.ent.mean_bytes(b) if int(b) in self.ent.nlog2p else float(b)) for b in R]
        if any(c[i] >= c[i + 1] for i in range(len(c) - 1)):
            print(f"[warn] entropy-coded level costs {c} are not increasing; allocating with nominal sizes")
            return list(R)
        return c


# ------------------------------------------------------------------ classifier
def make_classifier(cfg):
    import torchvision
    c = cfg["classifier"]
    if c["arch"] == "resnet50":
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if (c.get("imagenet_init") and not cfg["data"].get("synthetic")) else None
        m = torchvision.models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, cfg["data"]["num_classes"])
    elif c["arch"] == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if (c.get("imagenet_init") and not cfg["data"].get("synthetic")) else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, cfg["data"]["num_classes"])
    elif c["arch"] == "hf_vit":
        return HFClassifier(cfg)
    else:
        raise ValueError(c["arch"])
    return Normalized(m)


class HFClassifier(nn.Module):
    """Receiver classifier of the submitted manuscript: the Hugging Face ViT fine-tuned on TinyImageNet
    (classifier.hf_model, default 'tzhao3/vit-tiny-imagenet'), as in the authors' original
    classification.py (AutoImageProcessor + AutoModelForImageClassification).

    The original script converted each reconstruction to a PIL image and called the processor.
    Here the same preprocessing (resize to the processor size with its resampling filter, rescale,
    normalize with its mean/std) is done on GPU tensors so that batches can be classified.
    Output logits are re-ordered to this package's class order (sorted wnids)."""

    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        name = cfg["classifier"].get("hf_model") or "tzhao3/vit-tiny-imagenet"
        self.m = AutoModelForImageClassification.from_pretrained(name).eval()
        proc = AutoImageProcessor.from_pretrained(name)
        size = proc.size

        def _get(k):  # dict (transformers 4.x) or SizeDict (5.x) or int
            if isinstance(size, int):
                return size
            try:
                v = size[k]
            except (KeyError, TypeError):
                v = getattr(size, k, None)
            return int(v) if v is not None else None

        if _get("height") is not None:
            self.hw = (_get("height"), _get("width"))
        else:
            se = _get("shortest_edge") or 224
            self.hw = (se, se)
        self.mode = {2: "bilinear", 3: "bicubic"}.get(int(getattr(proc, "resample", 2)), "bilinear")
        self.do_norm = bool(getattr(proc, "do_normalize", True))
        mean = getattr(proc, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(proc, "image_std", [0.5, 0.5, 0.5])
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        perm, how = hf_label_permutation(self.m.config.id2label, cfg)
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long))
        self.label_mapping = how
        self.id2label = {int(k): v for k, v in self.m.config.id2label.items()}
        # a calibrated mapping (scripts/s0c_check_classifier.py) overrides the name-based guess
        import os
        from .utils import out_path
        pp = out_path(cfg, "ckpt", "hf_label_perm.npy")
        if os.path.exists(pp):
            import numpy as np
            self.perm.copy_(torch.as_tensor(np.load(pp), dtype=torch.long))
            self.label_mapping = "calibrated on training images (ckpt/hf_label_perm.npy)"
        print(f"[HFClassifier] {name}: input {self.hw}, resample {self.mode}, label mapping: {how}")

    def raw_logits(self, x):
        x = torch.round(x.clamp(0, 1) * 255) / 255
        x = F.interpolate(x, size=self.hw, mode=self.mode, align_corners=False, antialias=True).clamp(0, 1)
        if self.do_norm:
            x = (x - self.mean) / self.std
        return self.m(pixel_values=x).logits

    @torch.no_grad()
    def calibrate(self, batches, n_classes):
        """Find the model-output index of every dataset class from labelled TRAINING images:
        confusion matrix (dataset class x model output) + Hungarian assignment (one-to-one).
        Needed when the model's id2label does not identify TinyImageNet wnids/names."""
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        conf = None
        for x, y in batches:
            pred = self.raw_logits(x).argmax(1).cpu().numpy()
            if conf is None:
                conf = np.zeros((n_classes, self.m.config.num_labels), dtype=np.int64)
            np.add.at(conf, (y.numpy(), pred), 1)
        rows, cols = linear_sum_assignment(-conf)
        perm = np.empty(n_classes, dtype=np.int64)
        perm[rows] = cols
        agree = conf[rows, cols].sum() / max(conf.sum(), 1)
        changed = int((torch.as_tensor(perm) != self.perm.cpu()).sum())
        self.perm.copy_(torch.as_tensor(perm, dtype=torch.long))
        self.label_mapping = "calibrated on training images (Hungarian assignment)"
        return perm, float(agree), changed

    def forward(self, x):
        x = torch.round(x.clamp(0, 1) * 255) / 255  # the original pipeline went through an 8-bit PIL image
        x = F.interpolate(x, size=self.hw, mode=self.mode, align_corners=False, antialias=True).clamp(0, 1)
        if self.do_norm:
            x = (x - self.mean) / self.std
        return self.m(pixel_values=x).logits[:, self.perm]


def hf_label_permutation(id2label, cfg):
    """perm[k] = index of the model output that corresponds to dataset class k (sorted wnids)."""
    import os
    import re
    n = cfg["data"]["num_classes"]
    labels = {int(k): str(v).strip() for k, v in id2label.items()}
    root = cfg["data"]["root"]
    wnids_path = os.path.join(root, "wnids.txt")
    if cfg["data"].get("synthetic") or not os.path.exists(wnids_path):
        return list(range(n)), "identity (no dataset metadata available)"
    wnids = sorted(l.strip() for l in open(wnids_path))
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()  # noqa: E731
    # 1) labels are wnids
    inv = {v: k for k, v in labels.items()}
    if all(w in inv for w in wnids):
        return [inv[w] for w in wnids], "wnid"
    # 2) labels are class names from words.txt (full line or first synonym)
    words = {}
    wp = os.path.join(root, "words.txt")
    if os.path.exists(wp):
        for line in open(wp):
            w, name = line.rstrip("\n").split("\t", 1)
            words[w] = name
    if words:
        for key in ("full", "first"):
            inv_n = {}
            for k, v in labels.items():
                inv_n.setdefault(norm(v), k)
            perm = []
            for w in wnids:
                nm = words.get(w, "")
                nm = nm if key == "full" else nm.split(",")[0]
                perm.append(inv_n.get(norm(nm)))
            if all(p is not None for p in perm) and len(set(perm)) == n:
                return perm, f"class names ({key} words.txt entry)"
    # 3) generic labels (LABEL_k / 0..199): assume the training order was the sorted wnids
    if len(labels) == n:
        return list(range(n)), "index order assumed = sorted wnids (generic labels) -- verify clean accuracy"
    raise ValueError(f"cannot map {len(labels)} classifier labels to the {n} TinyImageNet classes")


class Normalized(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.m((x - self.mean) / self.std)

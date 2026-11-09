"""TinyImageNet loading (Methods: Dataset specifications).

Every sample returns BOTH the native 64x64 image and the 320x480 network input, so that
rates can be reported per native pixel and the native-resolution baseline can be
evaluated on exactly the same images.
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def resize(x, hw, mode="bicubic"):
    """x: (B,3,h,w) in [0,1] -> (B,3,H,W) in [0,1]. Non aspect-preserving (as in the paper)."""
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    if mode == "bicubic":
        kw["antialias"] = True
    return F.interpolate(x, size=tuple(hw), mode=mode, **kw).clamp(0, 1)


class TinyImageNet(Dataset):
    def __init__(self, root, split="train", num=None, seed=0):
        self.root = root
        wnids = [l.strip() for l in open(os.path.join(root, "wnids.txt"))]
        self.cls = {w: i for i, w in enumerate(sorted(wnids))}
        items = []
        if split == "train":
            for w in sorted(wnids):
                d = os.path.join(root, "train", w, "images")
                for fn in sorted(os.listdir(d)):
                    items.append((os.path.join(d, fn), self.cls[w]))
        else:  # 'val' used as labelled test split
            ann = os.path.join(root, "val", "val_annotations.txt")
            for line in open(ann):
                fn, w = line.split("\t")[:2]
                items.append((os.path.join(root, "val", "images", fn), self.cls[w]))
        if num is not None and num < len(items):
            rng = random.Random(seed)
            items = rng.sample(items, num)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        im = Image.open(p).convert("RGB")
        x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        return x, y


class SyntheticTiny(Dataset):
    """Random structured images, only for --smoke runs (no dataset download needed)."""

    def __init__(self, n, num_classes=200, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 8, 8, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = F.interpolate(self.x[i:i + 1], size=(64, 64), mode="bilinear", align_corners=False)[0]
        return x.clamp(0, 1), int(self.y[i])


def get_dataset(cfg, split):
    d = cfg["data"]
    n = d["num_train"] if split == "train" else d["num_test"]
    if d.get("synthetic"):
        return SyntheticTiny(n or 16, d["num_classes"], seed=0 if split == "train" else 1)
    return TinyImageNet(d["root"], "train" if split == "train" else "val", num=n)


def get_loader(cfg, split, batch_size=None, shuffle=None):
    ds = get_dataset(cfg, split)
    return DataLoader(ds, batch_size=batch_size or cfg["train"]["batch_size"],
                      shuffle=(split == "train") if shuffle is None else shuffle,
                      num_workers=cfg["train"]["num_workers"], pin_memory=torch.cuda.is_available(),
                      drop_last=False)


def to_input(x_native, cfg):
    return resize(x_native, cfg["data"]["input_hw"], cfg["data"].get("interpolation", "bicubic"))


# ---------------- patch helpers ----------------
def patchify(x, p):
    """(B,3,H,W) -> (B, P, 3, p, p) with row-major 1D index (as in the paper)."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C, p, p)


def unpatchify(t, H, W):
    B, P, C, p, _ = t.shape
    t = t.reshape(B, H // p, W // p, C, p, p).permute(0, 3, 1, 4, 2, 5)
    return t.reshape(B, C, H, W)

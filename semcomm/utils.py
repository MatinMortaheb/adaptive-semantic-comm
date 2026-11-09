import json
import os
import random
import time

import numpy as np
import torch


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def device_of(cfg):
    d = cfg.get("device", "cuda")
    if d == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available -> CPU")
        d = "cpu"
    return torch.device(d)


def out_path(cfg, *parts):
    p = os.path.join(cfg["out_dir"], *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_ckpt(model, path, device="cpu"):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t


def limited(loader, max_steps):
    for i, b in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        yield b

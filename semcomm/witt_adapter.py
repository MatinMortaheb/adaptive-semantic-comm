"""Thin adapter around the authors' public WITT implementation (https://github.com/KeYang8/WITT).

The repository has no license file, so it is NOT redistributed here: scripts/s7_witt.py clones it
at run time. At import we copy the source to <out_dir>/_witt_src and make two device-handling
patches so that it also runs on CPU (`x.get_device()` -> `x.device`; on CPU only, `.cuda()` is dropped). The network itself is unchanged.
"""
import os
import re
import shutil
import subprocess
import sys
import types


def prepare_source(cfg, cpu=False):
    from .utils import out_path
    clone = out_path(cfg, "_witt_repo")
    if not os.path.exists(os.path.join(clone, "net", "network.py")):
        subprocess.run(["git", "clone", "--depth", "1", cfg["witt"]["repo"], clone], check=True)
    src = out_path(cfg, "_witt_src")
    if os.path.exists(src):
        shutil.rmtree(src)
    shutil.copytree(clone, src, ignore=shutil.ignore_patterns(".git", "results", "*.png"))
    for root, _, files in os.walk(src):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                s = open(p).read()
                s2 = re.sub(r"(\w+)\.get_device\(\)", r"\1.device", s)
                if cpu:  # hard-coded .cuda() calls (e.g. Swin attention masks) -> stay on CPU
                    s2 = s2.replace(".cuda()", "")
                if s2 != s:
                    open(p, "w").write(s2)
    return src


def build_witt(cfg, C, device):
    """Returns a WITT model for cfg.witt.input ('native' 64x64 or 'upsampled' 320x480)."""
    import torch
    import torch.nn as nn
    src = prepare_source(cfg, cpu=device.type != "cuda")
    if src not in sys.path:
        sys.path.insert(0, src)
    for k in [k for k in sys.modules if k == "net" or k.startswith("net.") or k == "loss" or k.startswith("loss.")]:
        del sys.modules[k]
    from net.network import WITT  # noqa

    native = cfg["witt"]["input"] == "native"
    hw = tuple(cfg["data"]["native_hw"] if native else cfg["data"]["input_hw"])
    args = types.SimpleNamespace(distortion_metric="MSE", trainset="CIFAR10" if native else "DIV2K",
                                 channel_type="awgn", multiple_snr=",".join(map(str, cfg["witt"]["train_snrs"])),
                                 model="WITT", C=C)

    class config:  # mirrors WITT/train.py
        pass_channel = True
        CUDA = device.type == "cuda"
        norm = False
        logger = None

    config.device = device
    if native:  # CIFAR-style configuration (small images)
        config.downsample = 2
        config.encoder_kwargs = dict(img_size=hw, patch_size=2, in_chans=3, embed_dims=[128, 256], depths=[2, 4],
                                     num_heads=[4, 8], C=C, window_size=2, mlp_ratio=4., qkv_bias=True,
                                     qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
        config.decoder_kwargs = dict(img_size=hw, embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=C,
                                     window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                     norm_layer=nn.LayerNorm, patch_norm=True)
    else:  # DIV2K-style configuration; window 10 so that 20x30 / 40x60 / 80x120 / 160x240 grids are divisible
        config.downsample = 4
        config.encoder_kwargs = dict(img_size=hw, patch_size=2, in_chans=3, embed_dims=[128, 192, 256, 320],
                                     depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=C, window_size=10,
                                     mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                                     patch_norm=True)
        config.decoder_kwargs = dict(img_size=hw, embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
                                     num_heads=[10, 8, 6, 4], C=C, window_size=10, mlp_ratio=4., qkv_bias=True,
                                     qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True)
    net = WITT(args, config).to(device)
    return net, hw


def channel_uses(C, hw, downsample):
    """complex channel uses per image = C * (H/2^d) * (W/2^d) / 2."""
    return C * (hw[0] // 2 ** downsample) * (hw[1] // 2 ** downsample) / 2

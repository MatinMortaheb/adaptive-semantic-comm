"""Shared setup for all scripts."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from semcomm.config import add_common_args, apply_smoke, load_config  # noqa: E402
from semcomm.utils import device_of, seed_all  # noqa: E402


def setup(description, extra=None):
    ap = add_common_args(argparse.ArgumentParser(description=description))
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    seed_all(0)
    return args, cfg, device_of(cfg)

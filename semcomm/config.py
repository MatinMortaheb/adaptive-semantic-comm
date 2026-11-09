"""Configuration loading. All experiment settings live in configs/default.yaml.

Values in the YAML mirror the paper (Methods: Model and hyperparameters) unless
marked `# ASSUMPTION`
(not stated in the paper; please check against your original code).
"""
import copy
import os
import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k, v):
        self[k] = v


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    if path and os.path.abspath(path) != os.path.join(here, "configs", "default.yaml"):
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        # key.sub=value
        k, v = ov.split("=", 1)
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = yaml.safe_load(v)
    if not os.path.isabs(cfg["out_dir"]):
        cfg["out_dir"] = os.path.join(here, cfg["out_dir"])
    return Cfg(cfg)


def add_common_args(parser):
    parser.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny synthetic run on CPU to check the pipeline end to end")
    return parser


def apply_smoke(cfg):
    """Shrink everything so a full script runs in ~a minute on CPU with random weights."""
    cfg["data"]["synthetic"] = True
    cfg["data"]["num_train"] = 16
    cfg["data"]["num_test"] = 8
    cfg["data"]["num_classes"] = 200
    cfg["vit"]["pretrained"] = False
    cfg["train"]["batch_size"] = 4
    cfg["train"]["num_workers"] = 0
    for k in ("vit_epochs", "classifier_epochs", "decoder_epochs", "codec_epochs", "witt_epochs"):
        cfg["train"][k] = 1
    cfg["train"]["max_steps_per_epoch"] = 2
    cfg["eval"]["seeds"] = [0]
    cfg["eval"]["bootstrap"] = 50
    cfg["out_dir"] = os.path.join(cfg.get("out_dir", "outputs"), "_smoke")
    cfg["data"]["input_hw"] = [80, 120]   # smaller input so the smoke test fits in CPU memory
    cfg["device"] = "cpu"
    return cfg

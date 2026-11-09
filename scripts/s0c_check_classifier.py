"""S0c - Check (and if needed calibrate) the receiver classifier before any evaluation (v3.1).

1) prints the classifier's own labels (id2label) and the name-based mapping to TinyImageNet;
2) with --calibrate (default for hf_vit): runs the classifier on labelled TRAINING images, builds the
   confusion matrix (dataset class x model output) and finds the one-to-one label mapping with the
   Hungarian algorithm; saves outputs/ckpt/hf_label_perm.npy, which every later script loads;
3) reports clean accuracy on the TEST split: native 64x64 image and the 320x480 pipeline input.
Expected: about the full-resolution accuracy of the submitted paper (about 84%)."""
from _common import setup

import numpy as np
import torch

from semcomm.data import get_loader, to_input
from semcomm.system import load_classifier
from semcomm.utils import limited, out_path, save_json


def extra(ap):
    ap.add_argument("--no_calibrate", action="store_true")
    ap.add_argument("--calib_images", type=int, default=10000, help="training images used for calibration")


def main():
    args, cfg, dev = setup(__doc__, extra)
    clf = load_classifier(cfg, dev)
    info = {}
    if hasattr(clf, "id2label"):
        ex = list(clf.id2label.items())[:8]
        print("model labels (first 8):", ex, "... total", len(clf.id2label))
        info["id2label_examples"] = [str(v) for _, v in ex]
        if not args.no_calibrate:
            cal_cfg = dict(cfg)
            cal_cfg["data"] = dict(cfg["data"], num_train=args.calib_images)
            dl = get_loader(cal_cfg, "train", batch_size=cfg["eval"]["batch_size"], shuffle=False)
            gen = ((x.to(dev), y) for x, y in limited(dl, cfg["train"]["max_steps_per_epoch"]))
            perm, agree, changed = clf.calibrate(gen, cfg["data"]["num_classes"])
            np.save(out_path(cfg, "ckpt", "hf_label_perm.npy"), perm)
            print(f"calibration: train accuracy under the assignment {agree:.4f}; "
                  f"{changed} of {len(perm)} classes differ from the name-based guess")
            info.update(calib_train_acc=agree, calib_changed=changed)
    dl = get_loader(cfg, "test", batch_size=cfg["eval"]["batch_size"], shuffle=False)
    c_nat = c_in = n = 0
    with torch.no_grad():
        for x, y in limited(dl, cfg["train"]["max_steps_per_epoch"]):
            x = x.to(dev)
            c_nat += (clf(x).argmax(1).cpu() == y).sum().item()
            c_in += (clf(to_input(x, cfg)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    res = {"arch": cfg["classifier"]["arch"], "label_mapping": getattr(clf, "label_mapping", "n/a"),
           "acc_native_64": c_nat / max(n, 1), "acc_320x480": c_in / max(n, 1), "n": n, **info}
    save_json(res, out_path(cfg, "results", "s0c_classifier_check.json"))
    print(res)


if __name__ == "__main__":
    main()

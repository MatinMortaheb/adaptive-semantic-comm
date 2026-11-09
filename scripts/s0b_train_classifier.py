"""S0b - Train the downstream classifier on clean TinyImageNet images at 320x480.
Skip if you have the classifier used in the paper (set classifier.ckpt in the config)."""
from _common import setup

import torch
import torch.nn.functional as F

from semcomm.data import get_loader, to_input
from semcomm.models import make_classifier
from semcomm.utils import limited, out_path, save_ckpt, save_json
from s0_finetune_vit import evaluate


def main():
    args, cfg, dev = setup(__doc__)
    tr = cfg["train"]
    clf = make_classifier(cfg).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=tr["classifier_lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["classifier_epochs"])
    scaler = torch.amp.GradScaler(dev.type, enabled=tr["amp"] and dev.type == "cuda")
    train_dl, test_dl = get_loader(cfg, "train"), get_loader(cfg, "test", shuffle=False)
    acc = 0.0
    for ep in range(tr["classifier_epochs"]):
        clf.train()
        for x, y in limited(train_dl, tr["max_steps_per_epoch"]):
            x, y = to_input(x.to(dev), cfg), y.to(dev)
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            with torch.autocast(dev.type, enabled=scaler.is_enabled()):
                loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        acc = evaluate(clf, test_dl, cfg, dev)
        print(f"epoch {ep}: test acc {acc:.4f}")
    save_ckpt(clf, out_path(cfg, "ckpt", "classifier.pt"))
    save_json({"test_acc_clean": acc}, out_path(cfg, "results", "s0b_classifier.json"))


if __name__ == "__main__":
    main()

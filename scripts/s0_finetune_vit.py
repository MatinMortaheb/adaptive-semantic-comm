"""S0 - Fine-tune the DINO-initialized ViT-S/8 on TinyImageNet at 320x480 with cross-entropy
(Algorithm 1, encoder training; clarifies). Skip if you already have the paper's checkpoint
(set vit.ckpt in the config)."""
from _common import setup

import torch
import torch.nn.functional as F

from semcomm.data import get_loader, to_input
from semcomm.models import ViTWithHead
from semcomm.utils import limited, out_path, save_ckpt, save_json


def main():
    args, cfg, dev = setup(__doc__)
    tr = cfg["train"]
    vit = ViTWithHead(cfg).to(dev)
    opt = torch.optim.AdamW(vit.parameters(), lr=tr["vit_lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["vit_epochs"])
    scaler = torch.amp.GradScaler(dev.type, enabled=tr["amp"] and dev.type == "cuda")
    train_dl, test_dl = get_loader(cfg, "train"), get_loader(cfg, "test", shuffle=False)
    for ep in range(tr["vit_epochs"]):
        vit.train()
        for x, y in limited(train_dl, tr["max_steps_per_epoch"]):
            x, y = to_input(x.to(dev), cfg), y.to(dev)
            with torch.autocast(dev.type, enabled=scaler.is_enabled()):
                loss = F.cross_entropy(vit(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        acc = evaluate(vit, test_dl, cfg, dev)
        print(f"epoch {ep}: loss {loss.item():.3f} test acc {acc:.4f}")
    save_ckpt(vit, out_path(cfg, "ckpt", "vit_finetuned.pt"))
    save_json({"test_acc": acc}, out_path(cfg, "results", "s0_vit.json"))


@torch.no_grad()
def evaluate(model, dl, cfg, dev):
    model.eval()
    c = n = 0
    for x, y in limited(dl, cfg["train"]["max_steps_per_epoch"]):
        p = model(to_input(x.to(dev), cfg)).argmax(1).cpu()
        c += (p == y).sum().item()
        n += len(y)
    return c / max(n, 1)


if __name__ == "__main__":
    main()

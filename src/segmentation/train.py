"""Training and evaluation. run(), evaluate2() and ARCH copied from notebooks/01_crack_segmentation_deepcrack.ipynb.

The notebook uses module-level globals (train, val, test, mk, dev, OUT, bce, dice_loss). They are kept as
module globals here; call set_split() before run().
"""
import json, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from .data import DS, DS2
from .losses import cldice

dev = "cuda"
OUT = "/content/drive/MyDrive/ntnu/runs"
bce, dice_loss = torch.nn.BCEWithLogitsLoss(), smp.losses.DiceLoss("binary")
mk = lambda items, aug, sh: torch.utils.data.DataLoader(DS(items, aug), batch_size=8, shuffle=sh, num_workers=2)
train = val = test = None


def set_split(tr, va, te):
    global train, val, test
    train, val, test = tr, va, te


# The notebook also has evaluate(loader), the day-1 variant that reads a global `model`; run() uses evaluate2.
@torch.no_grad()
def evaluate2(model, loader):
    model.eval(); ious, dices = [], []
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
        for a, b in zip(p, y):
            i = (a * b).sum().item(); u = ((a + b) > 0).sum().item(); s = a.sum().item() + b.sum().item()
            ious.append(i / (u + 1e-8)); dices.append(2 * i / (s + 1e-8))
    return float(np.mean(ious)), float(np.mean(dices))


ARCH = {"Unet": smp.Unet, "UnetPlusPlus": smp.UnetPlusPlus, "DeepLabV3Plus": smp.DeepLabV3Plus, "FPN": smp.FPN}


def run(arch, enc, loss_name, seed, epochs=40, bs=8, lr=3e-4):
    os.makedirs(OUT, exist_ok=True)
    tag = f"{arch}_{enc}_{loss_name}_s{seed}"
    if os.path.exists(f"{OUT}/{tag}.json"): return json.load(open(f"{OUT}/{tag}.json"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tl = DataLoader(DS2(train, True), batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    vl, te = mk(val, False, False), mk(test, False, False)
    model = ARCH[arch](enc, encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl))
    scaler = torch.amp.GradScaler("cuda"); best, ck = -1, f"{OUT}/{tag}.pt"
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16): out = model(x)
            out = out.float(); loss = bce(out, y) + dice_loss(out, y)
            if "cldice" in loss_name: loss = loss + 0.5 * cldice(out, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        vi, vd = evaluate2(model, vl)
        if vi > best: best = vi; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck)); ti, td = evaluate2(model, te)
    r = dict(tag=tag, arch=arch, enc=enc, loss=loss_name, seed=seed, val_iou=best, test_iou=ti, test_dice=td)
    json.dump(r, open(f"{OUT}/{tag}.json", "w")); print(r); return r

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainSet, ValSet, find_cases, split_cases
from model import UNet3D

REGIONS = ["WT", "TC", "ET"]


def soft_dice_loss(logits, target, eps=1e-5):
    p = torch.sigmoid(logits)
    dims = (0, 2, 3, 4)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def loss_fn(logits, target):
    return soft_dice_loss(logits, target) + F.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def dice_per_region(logits, target, eps=1e-5):
    pred = (torch.sigmoid(logits) > 0.5).float()
    dims = (2, 3, 4)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return (2 * inter + eps) / (denom + eps)  # (B,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder per patient")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--cache", default=None, help="folder made by preprocess.py")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    cases = find_cases(args.data)
    train_cases, val_cases = split_cases(cases)
    print(f"{len(cases)} patients -> {len(train_cases)} train / {len(val_cases)} val ({dev})")

    p = (args.patch,) * 3
    train_dl = DataLoader(TrainSet(train_cases, p, cache=args.cache), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True, pin_memory=True)
    val_dl = DataLoader(ValSet(val_cases, cache=args.cache), batch_size=1, num_workers=args.workers)

    model = UNet3D(4, 3, args.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {ep}/{args.epochs}", leave=False):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()

        model.eval()
        scores = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                    logits = model(x)
                scores.append(dice_per_region(logits.float(), y).cpu())
        d = torch.cat(scores).mean(0)
        mean = d.mean().item()
        print(f"ep {ep:3d} | loss {total / len(train_dl):.4f} | "
              + " ".join(f"{r} {v:.3f}" for r, v in zip(REGIONS, d.tolist()))
              + f" | mean {mean:.3f}")
        if mean > best:
            best = mean
            torch.save({"model": model.state_dict(), "base": args.base},
                       os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
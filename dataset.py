import glob
import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODS = ["t1", "t1ce", "t2", "flair"]
ET_LABEL = 4  # BraTS 2018-2021 use 4 for enhancing tumor; BraTS 2023 uses 3


def find_cases(root):
    """Each patient is a folder containing *_t1.nii.gz, *_t1ce, *_t2, *_flair, *_seg."""
    cases = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        files = {}
        for m in MODS + ["seg"]:
            hits = glob.glob(os.path.join(d, f"*_{m}.nii*"))
            if len(hits) != 1:
                break
            files[m] = hits[0]
        else:
            cases.append(files)
    return cases


def split_cases(cases, val_frac=0.2, seed=42):
    """Patient-level split (one folder = one patient), so no leakage."""
    cases = cases[:]
    random.Random(seed).shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    return cases[n_val:], cases[:n_val]


def load_case(files):
    imgs = []
    for m in MODS:
        x = nib.load(files[m]).get_fdata(dtype=np.float32)
        brain = x > 0
        if brain.any():  # z-score over brain voxels only
            x = (x - x[brain].mean()) / (x[brain].std() + 1e-8)
            x[~brain] = 0
        imgs.append(x)
    img = np.stack(imgs)  # (4,H,W,D)
    seg = np.rint(nib.load(files["seg"]).get_fdata()).astype(np.int16)
    wt = seg > 0
    tc = (seg == 1) | (seg == ET_LABEL)
    et = seg == ET_LABEL
    mask = np.stack([wt, tc, et]).astype(np.float32)  # (3,H,W,D)
    return img, mask


def pad_to(arr, shape):
    pads = [(0, 0)] + [(0, max(0, s - d)) for s, d in zip(shape, arr.shape[1:])]
    return np.pad(arr, pads)


class TrainSet(Dataset):
    def __init__(self, cases, patch=(128, 128, 128), samples_per_epoch=None, fg_prob=0.67):
        self.cases, self.patch, self.fg_prob = cases, patch, fg_prob
        self.n = samples_per_epoch or len(cases)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, mask = load_case(self.cases[i % len(self.cases)])
        img, mask = pad_to(img, self.patch), pad_to(mask, self.patch)
        shape = img.shape[1:]
        # choose crop centre: on tumour (fg_prob) or random
        if random.random() < self.fg_prob and mask[0].any():
            idx = np.argwhere(mask[0] > 0)
            centre = idx[random.randrange(len(idx))]
        else:
            centre = [random.randrange(s) for s in shape]
        sl = []
        for c, p, s in zip(centre, self.patch, shape):
            start = int(np.clip(c - p // 2, 0, s - p))
            sl.append(slice(start, start + p))
        img, mask = img[(slice(None), *sl)], mask[(slice(None), *sl)]
        # augmentation: random flips
        for ax in (1, 2, 3):
            if random.random() < 0.5:
                img, mask = np.flip(img, ax), np.flip(mask, ax)
        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())


class ValSet(Dataset):
    """Full volume, padded so every dimension is divisible by 16 (4 pooling steps)."""

    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        img, mask = load_case(self.cases[i])
        target = [int(np.ceil(s / 16) * 16) for s in img.shape[1:]]
        return torch.from_numpy(pad_to(img, target)), torch.from_numpy(pad_to(mask, target))

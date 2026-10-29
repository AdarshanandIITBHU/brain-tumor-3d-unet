import glob
import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODS = ["t1", "t1ce", "t2", "flair"]
ET_LABEL = 4  # BraTS 2018-2021 use 4 for enhancing tumour; BraTS 2023 uses 3


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


def case_name(files):
    return os.path.basename(os.path.dirname(files["t1"]))


def split_cases(cases, val_frac=0.2, seed=42):
    """Patient-level split (one folder = one patient), so no leakage."""
    cases = cases[:]
    random.Random(seed).shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    return cases[n_val:], cases[:n_val]


def seg_to_mask(seg):
    """Raw BraTS labels -> 3 overlapping binary channels: WT, TC, ET."""
    wt = seg > 0
    tc = (seg == 1) | (seg == ET_LABEL)
    et = seg == ET_LABEL
    return np.stack([wt, tc, et]).astype(np.float32)


def load_raw(files):
    """Read the 4 modalities (z-scored over brain voxels) and the raw label volume."""
    imgs = []
    for m in MODS:
        x = nib.load(files[m]).get_fdata(dtype=np.float32)
        brain = x > 0
        if brain.any():
            x = (x - x[brain].mean()) / (x[brain].std() + 1e-8)
            x[~brain] = 0
        imgs.append(x)
    img = np.stack(imgs)  # (4,H,W,D)
    seg = np.rint(nib.load(files["seg"]).get_fdata()).astype(np.int16)
    return img, seg


def load_case(files):
    img, seg = load_raw(files)
    return img, seg_to_mask(seg)


# ---------- cache (run preprocess.py once; makes training much faster) ----------
def build_cache(files, out_dir):
    """Crop to the brain bounding box, save img as float16 and seg as uint8."""
    img, seg = load_raw(files)
    nz = (img != 0).any(0) | (seg > 0)
    idx = np.argwhere(nz)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    name = case_name(files)
    np.save(os.path.join(out_dir, f"{name}_img.npy"), img[(slice(None), *sl)].astype(np.float16))
    np.save(os.path.join(out_dir, f"{name}_seg.npy"), seg[sl].astype(np.uint8))


def get_case(files, cache):
    """Returns (img, seg); with a cache these are memory-mapped (cheap to crop)."""
    if cache:
        name = case_name(files)
        img = np.load(os.path.join(cache, f"{name}_img.npy"), mmap_mode="r")
        seg = np.load(os.path.join(cache, f"{name}_seg.npy"), mmap_mode="r")
        return img, seg
    return load_raw(files)


def pad_to(arr, shape):
    """Zero-pad the last 3 dims up to `shape` (works for 3D and 4D arrays)."""
    pads = [(0, 0)] * (arr.ndim - 3) + [(0, max(0, s - d)) for s, d in zip(shape, arr.shape[-3:])]
    return np.pad(arr, pads)


class TrainSet(Dataset):
    def __init__(self, cases, patch=(128, 128, 128), samples_per_epoch=None,
                 fg_prob=0.67, cache=None):
        self.cases, self.patch, self.fg_prob, self.cache = cases, patch, fg_prob, cache
        self.n = samples_per_epoch or len(cases)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, seg = get_case(self.cases[i % len(self.cases)], self.cache)
        if any(s < p for s, p in zip(seg.shape, self.patch)):  # rare: tiny volume
            img, seg = pad_to(np.asarray(img), self.patch), pad_to(np.asarray(seg), self.patch)
        shape = seg.shape
        # choose crop centre: on tumour (fg_prob) or random
        if random.random() < self.fg_prob and np.any(seg > 0):
            idx = np.argwhere(np.asarray(seg) > 0)
            centre = idx[random.randrange(len(idx))]
        else:
            centre = [random.randrange(s) for s in shape]
        sl = []
        for c, p, s in zip(centre, self.patch, shape):
            start = int(np.clip(c - p // 2, 0, s - p))
            sl.append(slice(start, start + p))
        sl = tuple(sl)
        img = np.asarray(img[(slice(None), *sl)], dtype=np.float32)
        mask = seg_to_mask(np.asarray(seg[sl]))
        # augmentation: random flips
        for ax in (1, 2, 3):
            if random.random() < 0.5:
                img, mask = np.flip(img, ax), np.flip(mask, ax)
        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())


class ValSet(Dataset):
    """Full volume, padded so every dimension is divisible by 16 (4 pooling steps)."""

    def __init__(self, cases, cache=None):
        self.cases, self.cache = cases, cache

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        img, seg = get_case(self.cases[i], self.cache)
        img = np.asarray(img, dtype=np.float32)
        mask = seg_to_mask(np.asarray(seg))
        target = [int(np.ceil(s / 16) * 16) for s in mask.shape[1:]]
        return (torch.from_numpy(np.ascontiguousarray(pad_to(img, target))),
                torch.from_numpy(np.ascontiguousarray(pad_to(mask, target))))
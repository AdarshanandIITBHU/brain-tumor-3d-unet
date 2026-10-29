"""Predict a tumour mask for one patient folder and save it as NIfTI (for 3D mesh / AR next)."""
import argparse
import glob
import os

import nibabel as nib
import numpy as np
import torch

from dataset import MODS, find_cases, load_case, pad_to
from model import UNet3D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="patient folder")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="pred_mask.nii.gz")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    model = UNet3D(4, 3, ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    files = find_cases(os.path.dirname(os.path.abspath(args.case)))
    files = [f for f in files if os.path.dirname(f["t1"]) == os.path.abspath(args.case)][0]
    img, _ = load_case(files)
    H, W, D = img.shape[1:]
    target = [int(np.ceil(s / 16) * 16) for s in (H, W, D)]
    x = torch.from_numpy(pad_to(img, target))[None].to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0].cpu().numpy()[:, :H, :W, :D]
    wt, tc, et = prob > 0.5
    # label map: 1 = edema/rest of WT, 2 = non-enhancing core, 4 = enhancing (BraTS style)
    seg = np.zeros((H, W, D), dtype=np.uint8)
    seg[wt] = 1
    seg[tc] = 2
    seg[et] = 4
    affine = nib.load(files["t1"]).affine
    nib.save(nib.Nifti1Image(seg, affine), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

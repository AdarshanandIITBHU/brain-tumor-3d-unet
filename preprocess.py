"""One-off: normalise + crop every patient and save to a fast cache.
Usage: python preprocess.py --data <BraTS train folder> --out /kaggle/working/cache
"""
import argparse
import os

from tqdm import tqdm

from dataset import build_cache, find_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only first N patients (testing)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = find_cases(args.data)[: args.limit]
    for files in tqdm(cases):
        build_cache(files, args.out)
    print(f"cached {len(cases)} patients in {args.out}")


if __name__ == "__main__":
    main()
"""Cache CLIP ViT-B/32 image features for one evaluation dataset of the Defense-Prefix paper.

Run from the root of the Defense-Prefix repository, one dataset per process
(the authors' attack generator seeds `random` once at import, and the authors
ran eval.py once per dataset, so one process per dataset reproduces their setup).

The dataset objects are the authors' own classes from `datasets/`, built with the
same arguments as the authors' `dataset.py`, so splits, labels, class names,
prompt templates and synthetic attacks are exactly theirs. The only difference
from `dataset.py` is that `download` is configurable (Stanford Cars can no longer
be downloaded through torchvision).

Features are computed exactly as in the authors' eval.py: fp16 CLIP image
features, L2-normalised. Evaluating any Defense-Prefix token on these cached
features gives the same predictions as eval.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

REAL_WORLD = {"rta-100", "paint", "disentangling"}
device = "cuda" if torch.cuda.is_available() else "cpu"


def build(name, preprocess, root="datasets", download=True):
    """Mirror of the authors' dataset.py with a configurable download flag."""
    if name == "imagenet":
        from datasets.ImageNetV2 import ImageNetValDataset
        return ImageNetValDataset(location=root, transform=preprocess)
    if name == "caltech":
        from datasets.caltech101 import Caltech101
        return Caltech101(root=root, transform=preprocess, download=download, make_typographic_dataset=True)
    if name == "pets":
        from datasets.oxford_pets import OxfordIIITPet
        return OxfordIIITPet(root=root, split="test", transform=preprocess, download=download, make_typographic_dataset=True)
    if name == "cars":
        from datasets.stanford_cars import StanfordCars
        return StanfordCars(root=root, split="test", transform=preprocess, download=False, make_typographic_dataset=True)
    if name == "flowers":
        from datasets.flowers102 import Flowers102
        return Flowers102(root=root, split="test", transform=preprocess, download=download, make_typographic_dataset=True)
    if name == "food":
        from datasets.food101 import Food101
        return Food101(root=root, split="test", transform=preprocess, download=download)
    if name == "aircraft":
        from datasets.fgvcaircraft import FGVCAircraft
        return FGVCAircraft(root=root, split="test", transform=preprocess, download=download)
    if name == "dtd":
        from datasets.dtd import DTD
        return DTD(root=root, split="test", transform=preprocess, download=download)
    if name == "eurosat":
        from datasets.eurosat import EuroSAT
        return EuroSAT(root=root, split="test", transform=preprocess, download=download)
    if name == "sun":
        from datasets.sun397 import SUN397
        return SUN397(root=root, split="test", transform=preprocess, download=False)
    if name == "paint":
        from datasets.paint_ta import PAINTDataset
        return PAINTDataset(root=root, transform=preprocess)
    if name == "disentangling":
        from datasets.disentangling_ta import DisentanglingDataset
        return DisentanglingDataset(root=root, transform=preprocess)
    if name == "rta-100":
        from datasets.rta100 import RTA100
        return RTA100(root=root, transform=preprocess)
    raise ValueError(name)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="folder for <dataset>.pt")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    t0 = time.time()
    data = build(args.dataset, preprocess, root=args.root)  # also generates the synthetic attack images (seed 300)
    prep_s = time.time() - t0

    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    real = args.dataset in REAL_WORLD
    clean, attack, labels = [], [], []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        for original_image, typographic_image, target in tqdm(loader, desc=args.dataset):
            f = model.encode_image(original_image.to(device))
            f /= f.norm(dim=-1, keepdim=True)          # identical to eval.py
            clean.append(f.cpu())
            if not real:
                g = model.encode_image(typographic_image.to(device))
                g /= g.norm(dim=-1, keepdim=True)
                attack.append(g.cpu())
            labels.append(torch.as_tensor(target))
    enc_s = time.time() - t1

    out = {
        "dataset": args.dataset,
        "real_world": real,
        "classes": list(data.classes),
        "template": data.templates[0],
        "labels": torch.cat(labels).long(),
        "clean": torch.cat(clean),                       # for real-world sets these images contain the attack
        "attack": torch.cat(attack) if attack else None,
        "meta": {
            "n_images": len(data),
            "n_classes": len(data.classes),
            "prepare_seconds": round(prep_s, 1),
            "encode_seconds": round(enc_s, 1),
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "torch": torch.__version__,
            "repo_commit": git_commit(),
            "model": "ViT-B/32",
        },
    }
    path = os.path.join(args.out, f"{args.dataset}.pt")
    torch.save(out, path)
    print(json.dumps({"saved": path, **out["meta"]}, indent=1))


if __name__ == "__main__":
    main()

"""Build a controlled seen/unseen attack-word test set and cache its CLIP features.

For every test image we render the SAME typographic attack twice (same font, colours,
starting font size and random position seed), once with a word the authors' DP vector
saw as an attack word during training (an ImageNet-100 class name, "seen") and once
with a length-matched word it never saw ("unseen", another ImageNet-1K class name that
shares no token with any ImageNet-100 name).  The clean image is cached too.

Rendering reuses the authors' generator (datasets/utils/make_dataset_train.py:
_transform + create_image, with the Pillow-10 textsize patch), so the attack style is
the paper's.  Evaluation candidates = dataset classes + all seen words + all unseen
words, so both conditions face exactly the same list of possible answers (the written
word is always among the candidates, as in the paper's real-world sets).

Usage (from the Defense-Prefix repo root):
  python make_seen_unseen.py --dataset pets --train_file /path/train_imagenet100.pt --out cache_su
"""
import argparse
import json
import os
import random
import re
import sys
import time

import io

import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import clip  # noqa: E402
from datasets.utils import make_dataset_train as gen  # noqa: E402  (authors' generator, patched)
from datasets.configs.imagenet_config import classes as IN1K_NAMES  # noqa: E402  (1000 CLIP-style names)

device = "cuda" if torch.cuda.is_available() else "cpu"
FONT_DIR = "datasets/font/"


def safe(name):
    """File-name-safe version of a class name (some names contain '/', e.g. 'shoji screen / room divider')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:40]


def tokens(name):
    return set(t for t in re.split(r"[^a-z0-9]+", name.lower()) if t)


def build_items(name, root):
    """Return (list of (image_path, label)), class names, template — from the authors' dataset classes."""
    if name == "pets":
        from datasets.oxford_pets import OxfordIIITPet
        d = OxfordIIITPet(root=root, split="test", download=True, make_typographic_dataset=False)
        items = list(zip([str(p) for p in d._images], d._labels))
    elif name == "caltech":
        from datasets.caltech101 import Caltech101
        d = Caltech101(root=root, download=True, make_typographic_dataset=False)
        items = [(os.path.join(d.root, "101_ObjectCategories", d.categories[d.original_y[i]],
                               f"image_{d.index[i]:04d}.jpg"), d.y[i]) for i in range(len(d.y))]
    else:
        raise ValueError(name)
    return items, list(d.classes), d.templates[0]


def word_pools(in100, dataset_classes, n_seed):
    ds = {c.lower() for c in dataset_classes}
    seen = [w for w in dict.fromkeys(in100) if w.lower() not in ds]
    in100_tokens = set().union(*[tokens(w) for w in in100])
    unseen_all = [w for w in dict.fromkeys(IN1K_NAMES)
                  if w.lower() not in ds and not (tokens(w) & in100_tokens) and len(w) <= 30]
    # length-match: for each seen word, take the unused unseen word with the closest length
    rng = random.Random(n_seed)
    rng.shuffle(unseen_all)
    unseen, used = [], set()
    for w in seen:
        best = min((u for u in unseen_all if u not in used), key=lambda u: abs(len(u) - len(w)))
        used.add(best)
        unseen.append(best)
    return seen, unseen


def render(path, word, seed):
    """Authors' rendering with a per-image seed; identical style for the seen and the unseen word."""
    r = random.Random(seed)
    font = os.path.join(FONT_DIR, r.choice(gen.font))
    fill = r.choice(gen.color)
    stroke = r.choice([c for c in gen.color if c != fill])
    img = gen._transform(Image.open(path))
    random.seed(seed)  # create_image draws font size and position from the global `random`
    img = gen.create_image(img, word, font, fill, stroke)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)  # the authors save attacked images as JPEG (quality=100) before encoding
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pets", "caltech"])
    ap.add_argument("--train_file", required=True, help="train_imagenet100.pt (for the 100 seen class names)")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--save_examples", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    in100 = list(torch.load(args.train_file, map_location="cpu", weights_only=False)["classes"])
    items, ds_classes, template = build_items(args.dataset, args.root)
    seen, unseen = word_pools(in100, ds_classes, args.seed)
    candidates = list(dict.fromkeys(ds_classes + seen + unseen))
    assert candidates[:len(ds_classes)] == ds_classes, "dataset class names must stay first (labels index them)"
    print(f"{args.dataset}: {len(items)} images | {len(ds_classes)} classes | {len(seen)} seen / {len(unseen)} unseen words")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    rng = random.Random(args.seed)
    picks = [rng.randrange(len(seen)) for _ in items]

    feats = {"clean": [], "seen": [], "unseen": []}
    ex_dir = os.path.join(args.out, f"examples_{args.dataset}")
    os.makedirs(ex_dir, exist_ok=True)
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for b in range(0, len(items), args.batch_size):
        batch = {"clean": [], "seen": [], "unseen": []}
        for i in range(b, min(b + args.batch_size, len(items))):
            path, _ = items[i]
            s = args.seed * 1_000_003 + i
            batch["clean"].append(preprocess(Image.open(path).convert("RGB")))  # as the authors' clean images
            im_s = render(path, seen[picks[i]], s)
            im_u = render(path, unseen[picks[i]], s)
            batch["seen"].append(preprocess(im_s))
            batch["unseen"].append(preprocess(im_u))
            if i < args.save_examples:
                im_s.save(os.path.join(ex_dir, f"{i:03d}_seen_{safe(seen[picks[i]])}.png"))
                im_u.save(os.path.join(ex_dir, f"{i:03d}_unseen_{safe(unseen[picks[i]])}.png"))
        with torch.no_grad():
            for k, ims in batch.items():
                f = model.encode_image(torch.stack(ims).to(device))
                feats[k].append((f / f.norm(dim=-1, keepdim=True)).cpu())
        print(f"  {min(b + args.batch_size, len(items))}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    out = {
        "dataset": args.dataset, "template": template, "candidates": candidates, "n_dataset_classes": len(ds_classes),
        "labels": torch.tensor([y for _, y in items]).long(),
        "clean": torch.cat(feats["clean"]), "seen": torch.cat(feats["seen"]), "unseen": torch.cat(feats["unseen"]),
        "word_seen": [seen[j] for j in picks], "word_unseen": [unseen[j] for j in picks],
        "pool_seen": seen, "pool_unseen": unseen,
        "meta": {"seed": args.seed, "n_images": len(items), "seconds": round(time.time() - t0, 1),
                 "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
                 "torch": torch.__version__},
    }
    path = os.path.join(args.out, f"su_{args.dataset}.pt")
    torch.save(out, path)
    with open(os.path.join(args.out, f"su_{args.dataset}_pools.json"), "w") as f:
        json.dump({"seen": seen, "unseen": unseen}, f, indent=1)
    print("saved", path, json.dumps(out["meta"]))


if __name__ == "__main__":
    main()

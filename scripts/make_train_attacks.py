"""Render two new attacked versions of every ImageNet-100 training image and cache their CLIP features.

For image i (fixed per-image seed) we draw ONE font, fill/stroke colour, font size and position, and
render the attack twice with the authors' generator (datasets/utils/make_dataset_train.py:
_transform + create_image, JPEG quality 100 round trip, exactly like their attacked images):
  ctrl  - a random wrong ImageNet-100 class name (the authors' rule)              -> renderer control
  vocab - with probability --p_extra an everyday word from vocab.json['v_train'],
          otherwise the same wrong ImageNet-100 class name as 'ctrl'              -> option B
So 'ctrl' and 'vocab' differ ONLY in the written word, and only on the images where an everyday word
was drawn. Rendering runs in DataLoader worker processes (the authors' generator is single-process);
the per-image seed makes the result independent of the number of workers.

Output (one file): relative paths (wnid/file), ImageNet-100 class name of each image, the two attack
words, fp16 L2-normalised features of both attacked versions. Clean features are NOT recomputed:
merge_train.py takes them from train_imagenet100.pt (matched by relative path).

Usage (from the Defense-Prefix repo root, on Kaggle):
  python make_train_attacks.py --src /kaggle/input/.../imagenet100 --vocab vocab.json --out /kaggle/working/cache
"""
import argparse
import io
import json
import os
import random
import sys
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.getcwd())
import clip  # noqa: E402
from datasets.utils import make_dataset_train as gen  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
FONT_DIR = "datasets/font/"


def render(img, word, seed, font, fill, stroke):
    random.seed(seed)  # create_image draws font size and position from the global `random`
    out = gen.create_image(img.copy(), word, font, fill, stroke)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=100)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class Attacks(Dataset):
    def __init__(self, items, preprocess):
        self.items, self.preprocess = items, preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, seed, w_ctrl, w_vocab, font, fill, stroke = self.items[i]
        base = gen._transform(Image.open(path))
        a = render(base, w_ctrl, seed, font, fill, stroke)
        b = a if w_vocab == w_ctrl else render(base, w_vocab, seed, font, fill, stroke)
        return self.preprocess(a), self.preprocess(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with Labels.json and train.X1..train.X4")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p_extra", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    labels = json.load(open(os.path.join(args.src, "Labels.json")))
    v_train = json.load(open(args.vocab))["v_train"]
    parts = sorted(d for d in os.listdir(args.src) if d.startswith("train.X"))
    rel, paths = [], []
    for part in parts:
        for wnid in sorted(os.listdir(os.path.join(args.src, part))):
            for f in sorted(os.listdir(os.path.join(args.src, part, wnid))):
                rel.append(f"{wnid}/{f}")
                paths.append(os.path.join(args.src, part, wnid, f))
    if args.limit:
        rel, paths = rel[:args.limit], paths[:args.limit]
    names = sorted({v.split(",")[0] for v in labels.values()})
    items, true_names, w_ctrl_all, w_vocab_all = [], [], [], []
    for i, (r, p) in enumerate(zip(rel, paths)):
        seed = args.seed * 1_000_003 + i
        rng = random.Random(seed)
        true = labels[r.split("/")[0]].split(",")[0]
        w_ctrl = rng.choice([n for n in names if n != true])
        w_vocab = rng.choice(v_train) if rng.random() < args.p_extra else w_ctrl
        font = os.path.join(FONT_DIR, rng.choice(gen.font))
        fill = rng.choice(gen.color)
        stroke = rng.choice([c for c in gen.color if c != fill])
        items.append((p, seed, w_ctrl, w_vocab, font, fill, stroke))
        true_names.append(true); w_ctrl_all.append(w_ctrl); w_vocab_all.append(w_vocab)
    n_extra = sum(a != b for a, b in zip(w_ctrl_all, w_vocab_all))
    print(f"{len(items)} images | everyday word on {n_extra} ({100 * n_extra / len(items):.1f}%) | workers {args.workers}")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    loader = DataLoader(Attacks(items, preprocess), batch_size=args.batch_size, num_workers=args.workers)
    f_ctrl, f_vocab = [], []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for k, (a, b) in enumerate(loader):
            fa, fb = model.encode_image(a.to(device)), model.encode_image(b.to(device))
            f_ctrl.append((fa / fa.norm(dim=-1, keepdim=True)).cpu())
            f_vocab.append((fb / fb.norm(dim=-1, keepdim=True)).cpu())
            if k % 50 == 0:
                print(f"  {min((k + 1) * args.batch_size, len(items))}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    out = {"relpaths": rel, "true_names": true_names, "word_ctrl": w_ctrl_all, "word_vocab": w_vocab_all,
           "style": [(os.path.basename(it[4]), it[5], it[6]) for it in items], "seeds": [it[1] for it in items],
           "typographic_ctrl": torch.cat(f_ctrl), "typographic_vocab": torch.cat(f_vocab), "v_train": v_train,
           "meta": {"p_extra": args.p_extra, "seed": args.seed, "n_images": len(items), "n_everyday": n_extra,
                    "seconds": round(time.time() - t0, 1), "workers": args.workers,
                    "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
                    "torch": torch.__version__}}
    path = os.path.join(args.out, "train_attacks_vocab.pt")
    torch.save(out, path)
    print("saved", path, json.dumps(out["meta"]))


if __name__ == "__main__":
    main()

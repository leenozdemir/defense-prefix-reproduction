"""Train a Defense-Prefix (DP) vector on cached CLIP image features.

Re-implements the authors' train.py (commit 50d7887) with one change: image features
are read from `train_imagenet100.pt` instead of being recomputed every step.  In the
authors' code the image features are computed under `torch.no_grad()` from a frozen
encoder, so caching them gives the same loss and the same gradients for the DP vector.

Kept identical to the authors' train.py:
  * CLIP ViT-B/32, loaded with clip.load (fp16 on GPU), frozen
  * DP vector: 1 x 512, initialised N(0, 0.02), inserted in place of the "*" token
    with the authors' encode_text_with_learnt_tokens (utils/non_nv.py)
  * one random template (of the 81 in templates.py) per batch, shared by all classes
  * loss = seta * CE(attacked images, DP class features) + gamma * KL(p_orig || p_dp) on clean images
    seta = 1, gamma = 3, logits scaled by model.logit_scale.exp()
  * SGD lr 0.002, CosineAnnealingLR(T_max=epochs, eta_min=5e-5) stepped once per epoch
  * batch size 512, 10 epochs, shuffled, last partial batch kept

Small numerical difference (recorded as a deviation): the CE and KL terms are computed in
fp32 from the fp16 logits (the authors compute them directly on fp16 logits).

Added (all optional, defaults reproduce the paper):
  --seed                 seeds python/numpy/torch and the shuffle order (the authors set no seed)
  --dp_dtype fp16|fp32   authors' DP parameter is fp16 (model.dtype); fp32 is offered as a deviation
  --no_defense / --no_reg  ablations: identity loss only / defense loss only
  --extra_words FILE     extra attack words added as distractor classes (for the broad-vocabulary run)
  If `typographic` in the training file has shape (K, N, 512), one of the K attacked
  versions is drawn per epoch (attack renewal).

Usage (from the Defense-Prefix repo root, so `utils/` and `templates.py` import):
  python train_dp_cached.py --train_file /path/train_imagenet100.pt --out runs/dp_seed0 --seed 0
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
import clip  # noqa: E402
from templates import imagenet_templates  # noqa: E402  (authors' 81 templates)
from utils.non_nv import encode_text_with_learnt_tokens  # noqa: E402  (authors' function)

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(d["classes"])
    labels = d["labels"].long()
    original = d["original"]
    typographic = d["typographic"]
    if typographic.dim() == 2:
        typographic = typographic.unsqueeze(0)  # (1, N, 512): a single fixed attacked version (authors' setup)
    assert original.shape[0] == labels.shape[0] == typographic.shape[1]
    return classes, labels, original, typographic, d.get("meta", {})


def text_features(model, prompts, emb=None, star=None):
    toks = clip.tokenize(prompts).to(device)
    if emb is None:
        f = model.encode_text(toks)
    else:
        f = model.encode_text_with_learnt_tokens(toks, star, emb, is_emb=True)
    return F.normalize(f, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--eta_min", type=float, default=0.00005)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--seta", type=float, default=1.0)
    ap.add_argument("--dp_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--no_defense", action="store_true", help="identity (KL) loss only")
    ap.add_argument("--no_reg", action="store_true", help="defense (CE) loss only")
    ap.add_argument("--extra_words", default=None, help="text file, one extra distractor word per line")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.encode_text_with_learnt_tokens = type(model.encode_text)(encode_text_with_learnt_tokens, model)
    star = clip.tokenize(["*"]).to(device)[0][1]

    classes, labels, original, typographic, train_meta = load_train(args.train_file)
    extra = []
    if args.extra_words:
        with open(args.extra_words) as f:
            extra = [w.strip() for w in f if w.strip() and w.strip() not in classes]
    candidates = classes + extra  # extra words can only be wrong answers
    n, k_versions = labels.shape[0], typographic.shape[0]
    print(f"train images {n} | classes {len(classes)} | extra distractors {len(extra)} | attacked versions {k_versions}")

    # DP vector, exactly like the authors' main(): N(0, 0.02) with the model's dtype unless fp32 is requested
    dtype = model.dtype if args.dp_dtype == "fp16" else torch.float32
    init = torch.empty(1, 512, dtype=dtype, device=device)
    nn.init.normal_(init, std=0.02)
    emb = nn.Embedding.from_pretrained(init, freeze=False)
    opt = torch.optim.SGD(emb.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.eta_min)

    logit_scale = model.logit_scale.exp().detach()
    gen = torch.Generator().manual_seed(args.seed)
    history, step = [], 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(args.epochs):
        version = int(torch.randint(k_versions, (1,), generator=gen)) if k_versions > 1 else 0
        order = torch.randperm(n, generator=gen)
        sums = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "acc_attack_dp": 0.0}
        nb = 0
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            x_clean = original[idx].to(device)
            x_typo = typographic[version, idx].to(device)
            y = labels[idx].to(device)

            template = random.choice(imagenet_templates)
            with torch.no_grad():
                t_orig = text_features(model, [template.format(c) for c in candidates])
            t_dp = text_features(model, [template.format("* " + c) for c in candidates], emb, star)

            logits_attack = logit_scale * x_typo @ t_dp.t()
            logits_orig = logit_scale * x_clean @ t_orig.t()
            logits_reg = logit_scale * x_clean @ t_dp.t()
            ce = F.cross_entropy(logits_attack.float(), y)
            kl = F.kl_div(F.log_softmax(logits_reg.float(), dim=-1), logits_orig.float().softmax(dim=-1),
                          reduction="batchmean")
            if args.no_defense:
                loss = kl
            elif args.no_reg:
                loss = ce
            else:
                loss = args.seta * ce + args.gamma * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            sums["loss"] += loss.item(); sums["ce"] += ce.item(); sums["kl"] += kl.item()
            sums["acc_attack_dp"] += (logits_attack.argmax(-1) == y).float().mean().item()
            nb += 1; step += 1
            if args.max_steps and step >= args.max_steps:
                break
        lr_used = sched.get_last_lr()[0]
        sched.step()
        rec = {"epoch": ep + 1, "lr": lr_used, "attack_version": version, "steps": nb,
               **{k: round(v / max(nb, 1), 5) for k, v in sums.items()}, "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))
        if args.max_steps and step >= args.max_steps:
            break

    token = emb.weight.detach().to(model.dtype).cpu()  # shape (1, 512), same as the authors' saved token
    torch.save(token, os.path.join(args.out, "dp_token.pt"))
    meta = {
        "args": vars(args), "train_meta": train_meta, "n_train": n, "n_candidates": len(candidates),
        "train_seconds": round(time.time() - t0, 1),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__, "history": history,
    }
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("saved", os.path.join(args.out, "dp_token.pt"), "| train seconds", meta["train_seconds"],
          "| peak MB", meta["peak_gpu_mem_mb"])


if __name__ == "__main__":
    main()

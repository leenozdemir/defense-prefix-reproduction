"""Combine the clean features of train_imagenet100.pt with the new attacked versions from make_train_attacks.py.

Writes two training files in the format train_dp_cached.py expects (classes, labels, original, typographic):
  train_ctrl.pt   attacked version 'ctrl'  (authors' word rule, our renderer)          -> renderer control
  train_vocab.pt  attacked version 'vocab' (50% everyday words from v_train)           -> option B
and v_train.txt (the everyday words, used as extra distractor classes with --extra_words).
Images are matched by relative path (wnid/file); class order and labels come from train_imagenet100.pt.

With --check it also measures, on the training images, how often PLAIN CLIP predicts the written word
(raw attack strength) for ImageNet-100 words vs everyday words — the premise of option B.

Usage:
  python merge_train.py --base .../train_imagenet100.pt --attacks .../train_attacks_vocab.pt --out .../cache_train --check
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--attacks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    att = torch.load(args.attacks, map_location="cpu", weights_only=False)
    base_rel = ["/".join(str(f).split("/")[-2:]) for f in base["files"]]
    pos = {r: i for i, r in enumerate(base_rel)}
    missing = [r for r in att["relpaths"] if r not in pos]
    assert not missing, f"{len(missing)} attacked images have no clean counterpart, e.g. {missing[:3]}"
    idx = torch.tensor([pos[r] for r in att["relpaths"]])
    classes = list(base["classes"])
    labels = base["labels"][idx]
    # sanity: the class name recorded at rendering time must equal the base label
    bad = sum(classes[int(l)] != t for l, t in zip(labels, att["true_names"]))
    assert bad == 0, f"{bad} label mismatches between the two files"
    for tag, key in (("ctrl", "typographic_ctrl"), ("vocab", "typographic_vocab")):
        torch.save({"classes": classes, "labels": labels, "original": base["original"][idx], "typographic": att[key],
                    "files": att["relpaths"], "meta": {**att["meta"], "variant": tag, "base": os.path.basename(args.base)}},
                   os.path.join(args.out, f"train_{tag}.pt"))
        print("saved", os.path.join(args.out, f"train_{tag}.pt"))
    with open(os.path.join(args.out, "v_train.txt"), "w") as f:
        f.write("\n".join(att["v_train"]))
    print(len(att["v_train"]), "everyday training words ->", os.path.join(args.out, "v_train.txt"))

    if args.check:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device)
        cand = classes + [w for w in att["v_train"] if w not in classes]
        with torch.no_grad():
            t = F.normalize(model.encode_text(clip.tokenize([f"a photo of a {c}." for c in cand]).to(device)), dim=-1)
            x = att["typographic_vocab"].to(device)
            pred = torch.cat([(x[i:i + 8192] @ t.T).argmax(-1).cpu() for i in range(0, len(x), 8192)])
        target = torch.tensor([cand.index(w) for w in att["word_vocab"]])
        everyday = torch.tensor([a != b for a, b in zip(att["word_ctrl"], att["word_vocab"])])
        hit = pred.eq(target).float()
        res = {"clip_attack_success_imagenet100_words": round(100 * hit[~everyday].mean().item(), 2),
               "clip_attack_success_everyday_words": round(100 * hit[everyday].mean().item(), 2),
               "n_imagenet100": int((~everyday).sum()), "n_everyday": int(everyday.sum())}
        print(json.dumps(res))
        json.dump(res, open(os.path.join(args.out, "attack_strength_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

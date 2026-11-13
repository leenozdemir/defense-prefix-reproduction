"""Build the 'everyday' attack vocabulary for the improvement (option B) and split it into train / held-out halves.

Source: the ImageNet-1K object, food and scene names (CLIP-style names, indices 398-999 of
datasets/configs/imagenet_config.py), i.e. everyday words such as 'coffee mug' or 'frying pan'.
In the seen/unseen diagnostic (notebook 04) these words fooled plain CLIP far more often than the
ImageNet-100 names the authors train on.

Removed (so no test set can contain a training attack word or a synonym of it):
  * any word sharing a token (>= 3 letters, not a stop word) with any class or attack name of the
    11 test sets we evaluate (data/test_set_class_names.json: Pets, Caltech, Flowers, Food, Aircraft,
    DTD, EuroSAT, Cars, PAINT, RTA-100, Disentangling). ImageNet-1K val itself is excluded from the
    option-B comparison, because its class names are the source of this vocabulary.
  * any word sharing a token with an ImageNet-100 training class name (so an attack word can never
    name or paraphrase the true class).
The remaining words are shuffled with a fixed seed: the first --n_train go to training, the rest are
held out for testing (never written on a training image, never used as a training distractor).

Usage (from the Defense-Prefix repo root):
  python build_vocab.py --labels /path/Labels.json --test_names data/test_set_class_names.json --out vocab.json
"""
import argparse
import json
import random
import re
import sys
import os

sys.path.insert(0, os.getcwd())
from datasets.configs.imagenet_config import classes as IN1K  # noqa: E402

STOP = {"a", "an", "the", "of", "and", "or", "with", "for", "in", "on", "to", "by"}
FIRST_OBJECT_INDEX = 398  # 'abacus': ImageNet-1K classes 0-397 are animals, 398-999 objects, food, scenes


def tokens(s):
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 3 and t not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ImageNet-100 Labels.json")
    ap.add_argument("--test_names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in100 = [v.split(",")[0] for v in json.load(open(args.labels)).values()]
    test_names = json.load(open(args.test_names))
    banned = set().union(*[tokens(n) for names in test_names.values() for n in names]) | \
        set().union(*[tokens(n) for n in in100])
    pool = [w for w in dict.fromkeys(IN1K[FIRST_OBJECT_INDEX:]) if len(w) <= 30 and not (tokens(w) & banned)]
    random.Random(args.seed).shuffle(pool)
    out = {"v_train": pool[:args.n_train], "v_test": pool[args.n_train:],
           "meta": {"source": "imagenet_config.classes[398:]", "n_pool": len(pool), "seed": args.seed,
                    "excluded_tokens_from": sorted(test_names) + ["imagenet100_train_classes"]}}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"pool {len(pool)} words -> {len(out['v_train'])} train / {len(out['v_test'])} held out")
    print("train examples:", out["v_train"][:12])
    print("held-out examples:", out["v_test"][:12])


if __name__ == "__main__":
    main()

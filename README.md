# Reproducing Defense-Prefix for CLIP: Word-General Robustness and the Limits of Attack-Vocabulary Diversification

Code, saved outputs and executed notebooks for the YZM5110 Multimodal Learning research project (Leen Özdemir).

- **Reproduced paper:** H. Azuma and Y. Matsui, *Defense-Prefix for Preventing Typographic Attacks on CLIP*, ICCV Workshops 2023 ([IEEE Xplore 10350494](https://ieeexplore.ieee.org/document/10350494), [arXiv 2304.04512](https://arxiv.org/abs/2304.04512)). Original code: [azuma164/Defense-Prefix](https://github.com/azuma164/Defense-Prefix), pinned at commit `50d7887017b295e76b15e56c137ab47e68264448`.
- **Report:** *Ozdemir_YZM5110_Research.pdf*, submitted separately through the course system. Its first page links to this repository.
- **Submitted version:** tag `v1.0-report`.

## What this repository shows

| Question | Result (CLIP ViT-B/32) |
|---|---|
| Does re-training Defense-Prefix (DP) reproduce the paper? | **Yes.** Means over 3 seeds: clean 61.21±0.15 / synthetic 44.97±0.40 / real-world 63.32±0.58, vs. 60.85 / 44.28 / 64.52 published on the same 12 test sets. |
| Is DP's robustness word-general or word-specific? | **Word-general.** On the same images, words DP never saw are not defended worse once their attack strength against plain CLIP is controlled. Word strength predicts which attacks survive. |
| Does diversifying the attack vocabulary in training help? | **No reliable gain.** Real-world +0.66 [−0.05, +1.34] (paired, 1,270 images); significant in 1 of 3 seeds. It costs 3.5× training time and 2.6× peak memory. |
| Controls | A random untrained token gives 13% (image-weighted) to 31% (3-set mean) of DP's real-world gain. The renderer and distractor controls are ≈0. |
| Failure analysis | After DP, most remaining real-world errors are not the written word (62% on RTA-100). |

## Repository layout

```
notebooks/             executed notebooks, in order (outputs kept as the record of each run)
  alternatives/        Colab versions of 06-2 and 07 (not executed; same code)
scripts/               every script used by the notebooks + analysis, table and figure scripts
data/                  class names of the test sets, the 200 added training words (v_train.txt)
outputs/runs/<run>/    learned tokens (dp_token.pt, 1x512 fp16) and train_log.json (time, memory, loss history)
outputs/results/       evaluation CSVs, per-image correctness, seen/unseen predictions, word-level analyses
outputs/failure_cases/ per-image predictions on RTA-100 / PAINT, example grids, summaries
outputs/env/           full pip freeze of the training environment
figures/               report figures (PNG + PDF), produced by scripts/make_figures.py
```

Run names: `dp_seed{0,1,2}` = reproduction; `vocab_seed{0,1,2}` = DP + diversified attack vocabulary (called "everyday words" in the notebooks); `ctrl_renderer`, `ctrl_distractors`, `ctrl_identity_only`, `ctrl_random_s{0..4}` = controls.

## Reproduce the tables and figures from the saved outputs (CPU, under a minute)

```bash
pip install numpy matplotlib          # torch is optional; scripts/ptload.py reads the .pt files without it
python scripts/analyze_word_groups.py  # -> outputs/results/groups06.json
python scripts/analyze_paired.py       # -> outputs/results/paired06.json
python scripts/make_tables.py          # -> report/tables/*.tex (the LaTeX tables of the report)
python scripts/make_figures.py         # -> figures/*.png, *.pdf  (fig4 also needs the authors' token, see below)
```

`make_figures.py` reads the authors' released token from `../ext/Defense-Prefix/learned_token/dp_vit-b32.pt`; set `AUTHORS_TOKEN=/path/to/dp_vit-b32.pt` if it is elsewhere.

## Re-running the experiments

All experiments ran on free **Colab** and **Kaggle** sessions with an **NVIDIA Tesla T4** (16 GB). Each notebook clones the authors' repository at the pinned commit and applies two compatibility patches:
1. `ImageDraw.textsize`, which was removed in Pillow 10, is replaced by `multiline_textbbox`.
2. `torch.load(..., weights_only=False)`.

It then writes our scripts into the repository root, because they import the authors' `datasets/`, `utils/` and `templates.py`.

| # | Notebook | Platform | What it does | Time (T4) |
|---|---|---|---|---|
| 01 | `01_verify_authors_eval_colab` | Colab | Original `eval.py` with the released token (RTA-100, Pets) | minutes |
| 02-1 | `02-1_prepare_test_sets_colab` | Colab | Builds 10 test sets with the authors' classes, renders synthetic attacks (seed 300), caches CLIP features | ~1 h |
| 02-2 | `02-2_prepare_extra_test_sets_kaggle` | Kaggle | Stanford Cars and ImageNet-1K val (the authors' download links are dead) | ~1 h |
| 03 | `03_prepare_train_set_kaggle` | Kaggle | ImageNet-100 train: 130,000 clean + attacked images, CLIP features | ~1.3 h |
| 04 | `04_seen_unseen_diagnostic_colab` | Colab | Seen vs. unseen attack words on the same images (Pets, Caltech101) | ~30 min |
| 05 | `05_retrain_and_evaluate_colab` | Colab | DP training on cached features, seeds 0–2, plus identity-only and random-token controls | 12.6 min / run |
| 06-1 | `06-1_render_everyday_attacks_kaggle` | Kaggle | Builds the added vocabulary (394 words → 200 train / 194 held out) and renders the diversified and control attacks | 13 min |
| 06-2 | `06-2_train_and_evaluate_everyday_kaggle` | Kaggle | Diversified-vocabulary runs (3 seeds), renderer and distractor controls, 5 random tokens, all evaluations | 45 min / run |
| 07 | `07_failure_cases_kaggle` | Kaggle (CPU) | Failure analysis on RTA-100 and PAINT (re-matches images to cached features) | ~5 min |

**Quick test (about 1 minute):** notebook 05, cell A4, trains for 5 steps and evaluates the token:
```bash
python train_dp_cached.py --train_file train_imagenet100.pt --out runs/smoke --seed 0 --max_steps 5
python eval_cached.py --cache cache_test --token runs/smoke/dp_token.pt --tag smoke --out results
```

**One full training run:**
```bash
python train_dp_cached.py --train_file train_imagenet100.pt --out runs/dp_seed0 --seed 0
python train_dp_cached.py --train_file train_vocab.pt --extra_words v_train.txt --out runs/vocab_seed0 --seed 0
```

**Results that need more resources:**
- Building the cached features (02-1, 02-2, 03) needs the raw datasets (≈20 GB of downloads) and about 3–4 GPU hours.
- Each diversified-vocabulary run needs about 5 GB of GPU memory.
- Everything after the caching step works on the cached features (≈0.6 GB) and fits a free T4 session.
- The cached features are not included here: they are derived from licensed datasets and are large.

## Environment

| | Colab | Kaggle |
|---|---|---|
| Python | 3.13.16 | 3.13.16 |
| torch / torchvision | 2.11.0 / 0.26.0 (later 2.14.1 / 0.29.1), cu130 | 2.11.0 / 0.26.0, cu130 |
| Pillow | 11.3.0 | 13.0.0 |
| CLIP | `openai/CLIP` commit `d05afc4` | same |
| other | `ftfy`, `regex`, `gdown`, `tqdm`, `numpy`, `matplotlib` | same |

`requirements.txt` pins the key packages. The complete environment of the training session (notebook 05, Colab) is in `outputs/env/pip_freeze_05_colab.txt`. A Colab update of torch from 2.11 to 2.14 gave identical evaluation numbers.

## Data access (licensed data is not redistributed)

| Data | Where to get it | Notes |
|---|---|---|
| ImageNet-100 (training) | Kaggle dataset `ambityga/imagenet100` | Source named by the authors; ImageNet terms of use (non-commercial research) |
| ImageNet-1K val | Kaggle competition `imagenet-object-localization-challenge` (join to accept the rules) | Rebuilt into the folder layout the authors' loader expects (notebook 02-2) |
| Caltech101, Oxford Pets, Flowers102, Food101, FGVC-Aircraft, DTD, EuroSAT | `torchvision` downloaders, through the authors' dataset classes | Caltech101 falls back to data.caltech.edu |
| Stanford Cars | Kaggle dataset `rickyyyyyyy/torchvision-stanford-cars` | The torchvision download is broken |
| RTA-100 | `rta100.zip` in the authors' repository | |
| PAINT real-world set | `typographic_attack_data.zip` in [mlfoundations/patching](https://github.com/mlfoundations/patching) | |
| Disentangling (Materzyńska et al.) | Google Drive folder linked from the [project page](https://joaanna.github.io/disentangling_spelling_in_clip/) | `classes.json` is not released; we use the 160 attacked images (see report, Table II) |
| SUN397 | not used | Download link dead |

## Deviations from the original setup

The full list, with consequences, is in the report (Table II). In short:
- SUN397 is missing.
- Disentangling uses 160 images instead of 171.
- `preprocessed_data` was rebuilt, because its code is not released.
- The Pillow patch above was applied.
- DP is trained on cached features. This is mathematically equivalent, because the images go through `no_grad`. The losses are computed in fp32.
- Seeds are set (the original sets none).
- Two issues in the original loaders were kept for comparability:
  - Caltech101 excludes the previous class name instead of the true one (off by one).
  - PAINT has 5 duplicated class names with a trailing dot.
- The real-world loaders use `os.listdir` and `list(set())`, so image and class order depend on the machine. Accuracy is unaffected; `failure_cases.py` re-matches images to features.

## Third-party code and licences

- The authors' Defense-Prefix code is **not copied** into this repository. It is cloned at run time at the pinned commit.
- Their `utils/non_nv.py` (from NVIDIA PALAVRA) is under the NVIDIA licence.
- OpenAI CLIP is MIT-licensed.
- Our own scripts and notebooks are released under the MIT licence (`LICENSE`).

## AI assistance

Claude (Anthropic) was used for search, coding and editing; the full disclosure is in the report.

## Citation of the reproduced work

```bibtex
@inproceedings{azuma2023defense,
  title     = {Defense-Prefix for Preventing Typographic Attacks on CLIP},
  author    = {Azuma, Hiroki and Matsui, Yusuke},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2023}
}
```

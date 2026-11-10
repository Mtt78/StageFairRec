# StageFairRec

PyTorch implementation of **Mitigating Sensitive Attribute Leakage in
LLM-Enhanced Course Recommendation via Stage-wise Fair Representation Learning**.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

The PyTorch command above is for the CPU demonstration. For GPU training use a
CUDA-compatible build from [PyTorch](https://pytorch.org/get-started/locally/).
For real LLM encoders and all attack classifiers:

```bash
python -m pip install -e '.[llm,attack,dev]'
```

Dependency ranges are in `pyproject.toml`. Record your resolved environment with
`python -m pip freeze > environment.txt` in each experiment archive. An example
validated environment is included in `docs/VALIDATION.md`; ranges are not a promise
that every future package version is compatible. No API key is required for local
model encoding. Download the model separately if the training machine is offline.

## Quick CPU demonstration

```bash
python -m pytest -q
python scripts/smoke.py --out runs/smoke
```

```bash
python scripts/smoke.py --out runs/smoke-sasrec --backbones sasrec
```

## data workflow

```bash
stagefairrec prepare --raw data/raw --out data/prepared --attribute gender

stagefairrec train --data data/prepared --config configs/default.json \
  --set backbone=sasrec --set variant=base --out runs/sasrec/base

stagefairrec encode --data data/prepared --mode sequential \
  --config configs/default.json --out cache/sequential

stagefairrec train --data data/prepared --config configs/default.json \
  --set backbone=sasrec --set variant=full --semantics cache/sequential \
  --backbone-checkpoint runs/sasrec/base/best.pt --out runs/sasrec/full

stagefairrec evaluate --data data/prepared --checkpoint runs/sasrec/full/best.pt \
  --semantics cache/sequential --split test --out runs/sasrec/full/test.json

stagefairrec export --data data/prepared --checkpoint runs/sasrec/full/best.pt \
  --semantics cache/sequential --out runs/sasrec/full/representations

stagefairrec attack --representations runs/sasrec/full/representations \
  --kinds mlp logistic random_forest xgboost --out runs/sasrec/full/attack.json
```

```bash
stagefairrec train --data data/prepared --config configs/default.json \
  --set backbone=sasrec --set variant=full --semantics cache/sequential \
  --backbone-checkpoint runs/sasrec/base/best.pt --out runs/sasrec/full \
  --resume runs/sasrec/full/last.pt
```

## Inference

Recommend from an existing cached prediction state using its integer row index in
`rows.npy`. This ranks the full catalog, separately from paper sampled evaluation:

```bash
stagefairrec recommend --data data/prepared --checkpoint runs/sasrec/full/best.pt \
  --semantics cache/sequential --row 11 --k 10 --exclude-seen \
  --out runs/sasrec/full/recommendations.json
```

A new history requires fresh semantic encoding; this CLI does not implement a live
serving API or cold-start learner registration. Sensitive labels are not supplied
to the scoring function.

## Release and attribution

The new implementation is released under the [MIT license](LICENSE).

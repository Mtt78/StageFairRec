# StageFairRec

Paper-based PyTorch implementation of **Mitigating Sensitive Attribute Leakage in
LLM-Enhanced Course Recommendation via Stage-wise Fair Representation Learning**.

**Status:** this repository reconstructs the method from the supplied manuscript.
It is not the original experiment repository, and does not claim to reproduce the
reported numbers. No real learner data, pretrained LLM weights, or paper-result
checkpoints are distributed. See [implementation decisions](docs/IMPLEMENTATION.md)
for every material assumption and [validation](docs/VALIDATION.md) for tests run.

[中文说明](README_zh.md) · [Data format](docs/DATA.md) · [Experiments](docs/EXPERIMENTS.md)

## What is implemented

- Frozen course-side text encoding and padding-aware hidden-state mean pooling.
- Learner-side LoRA, sensitive component subtraction and alternating source adversaries.
- User/course gated residual semantic fusion and post-fusion residual calibration.
- Target-conditioned local group consistency (TLGC) with same-target, opposite-group,
  similarity-thresholded top-K neighbors; an empty neighborhood gives zero loss.
- PMF, DeepModel, SASRec and BERT4Rec backbone implementations with explicit adaptations.
- Separate backbone pretraining and calibration with a frozen backbone.
- All Table 5 variants: Base+LLM, Source-only, Post-fusion Only, w/o TLGC, Full.
- Chronological per-user 8:1:1 split, fixed 1+99 ranking, HR/NDCG and group gaps.
- MLP, logistic regression, random forest and optional XGBoost leakage attacks.
- Checkpoints, resume, inference, three-seed experiment driver, aggregation and CI.

The original SM, FFVAE and AFRL baseline implementations are **not included**.
Those methods require their own repositories/protocol verification; they are not
replaced here with simplified methods under the same names.

## Install

Python 3.10+ is required. Create a virtual environment, install the PyTorch build
appropriate to your machine, then install this package from the repository root:

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

This generates synthetic CSVs, creates both cache modes, trains all four backbones
and five enhanced variants, ranks test candidates, exports representations, runs
an unseen-user logistic attacker, and returns full-catalog recommendations.
The toy encoder used here is **not an LLM**. Toy metrics do not measure paper performance.
Use a fresh output directory on each run. To run one backbone:

```bash
python scripts/smoke.py --out runs/smoke-sasrec --backbones sasrec
```

## Real-data workflow

Convert authorized source data into `courses.csv`, `users.csv`, `interactions.csv`
using the [explicit canonical schema](docs/DATA.md). No demographic label is guessed
from names, text, or history. Gender and age experiments use separately prepared datasets.

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

`encode` performs source training and writes both `plain.npy` (untuned encoder,
without source subtraction) and `fair.npy` (LoRA + subtraction). Courses always
use the frozen encoder with adapters disabled. Sequential caches are indexed by
prediction instance and use only the prefix; collaborative caches are indexed by
learner and use only training-observed histories. The cache mode must match the backbone.

For a local LLM use `--set model_name=/absolute/path/to/model`. Pin a model revision
with `--set revision=COMMIT_SHA`. Default targets are `q_proj` and `v_proj` for the
DeepSeek/Llama-style architecture. Other model families may require different module
names or explicit encoder layout adapters; do not assume a model-name substitution
alone reproduces the paper's encoder robustness results.

`best.pt` is chosen by validation HR@10 only; `last.pt` also contains optimizer and
RNG state. Calibration resumes in the same directory with the same config:

```bash
stagefairrec train --data data/prepared --config configs/default.json \
  --set backbone=sasrec --set variant=full --semantics cache/sequential \
  --backbone-checkpoint runs/sasrec/base/best.pt --out runs/sasrec/full \
  --resume runs/sasrec/full/last.pt
```

Resume is for an interrupted run with its original epoch budget, not a change to
hyperparameters or the dataset. Source-stage training currently restarts from a
fresh directory; resumable checkpoints apply to backbone/calibration training.

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

## Resource considerations

The manuscript uses a 7B LLM and an RTX 4090. This reconstruction does not guarantee
that every encoder/configuration fits a 24 GB GPU. The learner LLM uses LoRA and
optional gradient checkpointing, with a default source microbatch of 2. Course
features are frozen. Float32 prefix caches can be large: two caches of N states
and d_s features require about `2 * N * d_s * 4` bytes, excluding course features.
For 2 million states at d_s=4096, this is about 65.5 GB (61.0 GiB). Disk arrays are
memory mapped; they are not all moved to GPU.

## Release and attribution

The new implementation is released under the [MIT license](LICENSE); dataset and
LLM licenses remain separate. `CITATION.cff` intentionally contains collective
software authorship rather than invented manuscript bibliographic metadata.
Before an author release, replace it with the confirmed author/venue/DOI details,
verify the data preparation against your original experiments, and upload real
results only after rerunning them. See [GitHub publication steps](docs/GITHUB.md).

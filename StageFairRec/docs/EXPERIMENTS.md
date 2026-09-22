# Experiments

All commands run at repository root. Read `IMPLEMENTATION.md` before interpreting
results as reproduction evidence. Never select a checkpoint or hyperparameter by
test AUC/HR. Calibration uses validation HR@10 early stopping only.

## RQ1 and RQ3

| Variant name | Source | Residual | TLGC |
|---|---|---|---|
| `base` | no semantics | no | no |
| `llm` | no | no | no |
| `source_only` | yes | no | no |
| `post_only` | no | yes | yes |
| `wo_tlgc` | yes | yes | no |
| `full` | yes | yes | yes |

`post_only` vs `full` isolates the source intervention; `wo_tlgc` vs `full`
isolates TLGC. `llm` vs `source_only` measures source-only calibration.
All enhanced variants start from the same pretrained backbone checkpoint.
Plain and fair representations are generated together from the same model identity.

Create both cache modes for each seed:

```bash
for seed in 42 43 44; do
  for mode in collaborative sequential; do
    stagefairrec encode --data data/prepared --mode "$mode" \
      --set seed="$seed" --out "cache/$seed/$mode"
  done
done
python scripts/run_experiments.py --data data/prepared --cache-root cache --out runs/main
python scripts/aggregate.py --runs runs/main --out runs/main/summary.csv
```

Keep each dataset/attribute/encoder/parameter setting in separate output roots.
The aggregate script groups identical dataset hashes and refuses duplicate seeds.

## RQ2

This repository provides the proposed framework and Base/Base+LLM. It does not
supply SM, FFVAE or AFRL. To compare them, use verified implementations with the
same mappings, candidates and attack splits, and export their scores or representations
without pretending that an approximation is the original baseline.

## RQ4: validation-only sweeps

```bash
python scripts/sweep.py --data data/prepared --semantics cache/42/sequential \
  --backbone-checkpoint runs/main/42/sasrec/base/best.pt --out runs/lambda1 \
  --parameter lambda1 --values 0 0.01 0.05 0.1 0.2
```

Repeat for `lambda2`, keeping `lambda1=0.1`. The grid above is a configurable
example; it is not claimed to recover all original plotted settings. A sweep only
trains and writes validation results. Choose the configuration, then explicitly
run final test evaluation. Check `batches_with_tlgc` in training logs: small batches
and sparse target overlap may make TLGC inactive.

After fixing the recommender, use:

```bash
stagefairrec attack --representations runs/model/representations \
  --kinds mlp logistic random_forest xgboost --out runs/model/attack.json
stagefairrec attack --representations runs/model/representations \
  --kinds mlp logistic random_forest xgboost --protocol user_disjoint \
  --out runs/model/attack-unseen-users.json
```

Keep those protocols separate. Temporal default follows the manuscript split
wording but may share identities. Unseen-user partition is seeded stratified 60/20/20;
its different fractions are explicitly a reconstruction choice. Each temporal split
must contain every retained sensitive class or AUC evaluation fails.

## RQ5

Create a new cache for every LLM model/revision and random seed; retain all other
settings. Default DeepSeek/Llama-like models use `q_proj,v_proj`; Qwen models with
these modules can use the same interface. ChatGLM architectures with fused QKV or
sequence-first outputs require explicitly configured targets/layout adapters. No
untested encoder family is claimed as verified. A local model name can be used.
Do not reuse semantic caches across encoders or datasets. Cache/checkpoint hashes
are validated on load.

## Outputs and reproducibility

Each train run saves config, history, best/last checkpoints and final validation
metrics. Evaluation is an explicit command, so training never touches test scores.
Attack tuning records every validation configuration. Aggregation computes means
and sample standard deviations of actual saved metrics; the paper numbers are
not injected anywhere. Checkpoints record RNG/optimizer state for interrupted runs.
GPU floating-point kernels may still prevent bitwise equivalence across hardware.

# Validation record

Validation performed in a CPU Python 3.12 environment during package creation.

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.14.0+cpu |
| scikit-learn | 1.8.0 |
| Transformers | 4.57.6 |
| PEFT | 0.21.0 |
| Accelerate | 1.15.0 |

Test result: **15 passed** (including the optional local HF/PEFT test).

Executed checks:

- Mathematical unit tests: fixed candidate exclusion/identity, chronological prefix
  visibility, explicit age boundaries, protected existing outputs, all four
  backbone forwards/backwards, SASRec causal invariance, PFD gradient ownership,
  TLGC numerical value/empty neighborhoods/self-learner exclusion, masked mean
  pooling, HR/NDCG and group gaps.
- Optional actual Transformers/PEFT test with a locally instantiated tiny random
  Llama model and tokenizer: forward, backward, LoRA-only trainability, and frozen
  base output invariance when adapters are disabled. No pretrained weights downloaded.
- Binary and multiclass macro-OVR AUC checks and a stratified user-disjoint attack.
- End-to-end CPU synthetic run for PMF, DeepModel, SASRec and BERT4Rec: one base
  plus five enhanced variants each (24 training runs), two epochs each; 20 enhanced
  test evaluations; 4 representation exports; 4 logistic unseen-user audits;
  4 full-catalog inference commands. Both source cache modes built successfully.
- Python compilation and editable-package installation.

**Not validated:** a full 7B source-stage run, RTX 4090 memory/performance, original
MOOCCubeX/User-Activity snapshots, published table values, external fairness
baselines, all alternative pretrained encoder architectures, or GPU bitwise resume
behavior. Optional XGBoost depends on the separately installed extra and was not
part of the synthetic integration run. Those are real limitations, not implied passes.

The integration outputs are synthetic and are not distributed as paper results.
Re-run `python -m pytest -q` and `python scripts/smoke.py` to reproduce these checks.
If the optional LLM dependencies are absent, the local HF test is explicitly skipped.

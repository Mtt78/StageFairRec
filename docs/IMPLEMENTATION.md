# Manuscript mapping and reconstruction decisions

Source: the PDF supplied for this task, titled *Mitigating Sensitive Attribute
Leakage in LLM-Enhanced Course Recommendation via Stage-wise Fair Representation
Learning*. No original repository or experiment outputs were available to inspect.

## Equations

| Manuscript | Implementation | Notes |
|---|---|---|
| (2), course mean pooling | `semantics.HFEncoder`, `mean_pool` | Padding excluded, adapter disabled, frozen base |
| (3), learner LoRA pooling | `semantics.HFEncoder` | Chronological text; left token truncation |
| (4)–(6), source decomposition | `models.SourceFair`, `semantics.encode` | Alternating detached discriminator and encoder/extractor/classifier step |
| (7)–(9), projected gated fusion | `models.Fusion` | Single sigmoid vector gate; L2 normalization chosen |
| (10), residual subtraction | `models.StageFairRec.learner` | Sensitive label not needed at inference |
| (11)–(13), PFD | `engine.fit` | Separate discriminator update; PFD sees detached z |
| (14)–(16), TLGC | `losses.tlgc_loss` | Stop-gradient neighbor selection, same target, cross group, distinct learners |
| (17), backbone scoring | `models.Backbone.score` | Dot product; DeepModel includes nonlinear head |
| (18), source training | `semantics.encode` | Train-terminal prompts only; cache after fitting |
| (19), calibration | `engine.fit` | Backbone frozen, eval mode; recommendation + lambda1 PFD + lambda2 TLGC |
| (20), group gaps | `engine.summarize_ranks` | Max-minus-min group HR/NDCG, observed groups only |

## Choices not uniquely specified by the manuscript

These are implementation decisions, not recovered original experimental settings.

| Topic | Choice here | Reproduction implication |
|---|---|---|
| Dataset conversion | Explicit canonical CSV/JSONL converter | Actual raw schemas and author filtering must be reconciled |
| Age bins | Mandatory user-specified boundaries | Cannot infer published age groups |
| Duplicates / ties | Retain events; stable time then row order | Original deduplication policy unknown |
| Split rounding | floor(0.8n), floor(0.1n), remainder | Counts may differ from published processed data |
| Norm | L2 normalization | Manuscript does not identify LayerNorm vs L2 |
| Embedding width/sequence width | d=64, max_len=50, 2 layers/heads | Defaults chosen, not recovered |
| Source iterations | 5 epochs, fixed budget; one terminal prompt/user | Source stopping rule and sample schedule unspecified |
| Source lambda / beta | 0.1 / 0.1 | Chosen defaults |
| lambda1 / lambda2 | 0.1 / 0.1 | Reported balanced setting in Section 5.5 |
| TLGC | Kn=10, threshold=0, temperature=0.1 | Chosen defaults; inspect actual active-neighborhood frequency |
| Repeated batch learners | Select a distinct reference per learner; one row per learner in denominator, matched-target row preferred | Resolves set notation for repeated prefix samples; exact author sampler unknown |
| PFD gradient scope | PFD updates residual/extractor heads only, via detached z | Follows prose/explicit variables in Eq. (13); recommendation/TLGC train fusion |
| Source vocabulary | Deterministic sorted metadata fields, no demographics | Exact historical prompts unavailable |
| Source baseline cache | Untuned base encoder, adapters disabled | Isolates source intervention from LoRA effects |
| PMF | Dot product, sampled binary targets with MSE, optional Adam weight decay | Implicit-feedback adaptation, not verified original PMF objective |
| DeepModel | Wide dot product plus two-layer matching MLP over paired embeddings | Paper cites Wide & Deep but does not specify actual architecture/features |
| SASRec | Causal Transformer, prefix-to-next sampled BCE | Compact independent implementation; not byte-equivalent to official SASRec |
| BERT4Rec pretraining | Bidirectional Cloze, 80/10/10 corruption, full-item CE at masked positions | Explicit canonical adaptation; user sequence prefixes form training windows |
| BERT4Rec calibration | Last MASK state; sampled positive/negative CE | Necessary scoring-stage adaptation; differs from pretraining Cloze objective |
| Training negatives | Draw from all known unobserved courses with replacement | Future positives only excluded from negative pool, never added to prompts |
| Evaluation negatives | 99 distinct unseen courses, generated once | Fails if pool too small; never silently changes the paper protocol |
| Ranking ties | Pessimistic rank for tied negatives | Avoids placing target first because candidates store it first |
| Attack representation | Mean scoring state per learner per temporal split | Manuscript leaves representation aggregation unspecified |
| Attack temporal split | Same learners may occur across partitions | Collaborative embeddings may be identical; unseen-user audit additionally available |
| Attack tuning | Small specified validation grid; maximize validation AUC | Exact original grids were not supplied |
| Seed selection | 42, 43, 44 | Three repeats specified, original seeds unknown |
| Regularization | Adam weight decay=0, gradient clip=5 | Chosen defaults |

## Important interpretation limits

1. Representation leakage is measured by attacks, not a proven causal or privacy guarantee.
2. AUC near 0.5 is the reference; an AUC well below 0.5 can be inverted and does not
   mean stronger fairness. Reports retain raw AUC and distance from chance.
3. The manuscript's chronological representation split can share learner identities.
   The provided `user_disjoint` audit measures a different question and must be
   labeled separately. It is not a silent replacement of the stated protocol.
4. Fair-source training uses only training histories; sequential cache rows always
   exclude the target event and later events. Collaborative training semantics
   include training-observed targets by design and never validation/test interactions.
5. Manuscript statistics and performance values are not encoded as outputs or tests.
6. No SM/FFVAE/AFRL reimplementations are claimed; complete replication of Table 2–4
   comparisons needs their original implementations and matched training protocols.
7. This repository's model-agnostic interface supports the implemented scoring heads.
   It does not imply that all arbitrary recommendation architectures can be plugged in
   without writing a backbone adapter and verifying the original objective.

## Primary implementation references

The method equations come from the supplied manuscript. Library API usage follows
[PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora)
and [PEFT model configuration](https://huggingface.co/docs/peft/main/en/guides/peft_model_config).
All backbone/fairness code in this package was independently written for this task;
no third-party research repository source was copied.

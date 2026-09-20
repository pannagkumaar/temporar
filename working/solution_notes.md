# Antibody Light-Chain Pairing Reconstruction — solution notes

## Task
Per block: 8 heavy chains, 8 light chains, scrambled **bijection**. Submit a ranking of the 8
candidates per heavy row. `row_credit=(8-rank)/7`; `adjusted=clip(2*mean-1,0,1)`;
`score = 0.75*adjusted(all) + 0.25*min(adjusted(held_out_subject), adjusted(cross_study))`.
Test = 4096 rows / 512 blocks / 64 human samples (8 blocks per sample). Train = 30376 rows /
3797 blocks / 105 samples (94% human, rest mouse/rat).

## Incumbent
`feats3.FB3` + `model.BlockMatcher` (gene embeddings + bilinear + MLP over 115 pair features),
listwise row CE + 0.5 * column CE, Sinkhorn-in-the-loop during training, 3 seeds averaged,
**decoded with exact permutation marginals** (`permmarg.py`, T=1.25).
**OOF adjusted = 0.4850** on 5-fold sample-grouped CV (30376 rows). min-fold 0.4452.
Artifacts: `working/cache/oofraw_v3s3.npy`, `working/cache/oof_v3s3.npy`.

### Ledger (5-fold GroupKFold on sample_id, same folds throughout)
| step | OOF adj | delta |
|---|---|---|
| v1 features, Sinkhorn decode, 1 seed | 0.4628 | — |
| v2 (probabilistic surprisal SHM) | 0.4441 | -0.019 **rejected** |
| v2 + per-position residue towers | ~0.43 | **rejected**, train 0.76 / val 0.45 |
| v3 features, no LOO, 1 seed | 0.4673 | +0.005 |
| v3 + leave-one-out lift tables | 0.4758 | **+0.0085** |
| v3 + LOO + 3 seeds | 0.4794 | +0.0036 |
| ... + exact permutation marginals (vs as-trained Sinkhorn) | **0.4850** | **+0.0056** |

Decoder detail: raw logits 0.4633 / Sinkhorn as trained 0.4794 / Sinkhorn best T 0.4833 /
**exact marginals 0.4850**. Ranking by exact marginal posterior is provably optimal for a
rank-linear metric under a pairwise permutation model, and costs 2.2 s for 3797 blocks.

### Pretrained LM embeddings (frozen, mean-pooled, 5 folds x 3 seeds, marginal decode)
Single-model arms on CPU, then all arms re-run together on one GPU worker so the comparison is
like-for-like (the GPU base arm reproduces the CPU base arm: 0.4874 vs 0.4858).

| arm | OOF | human | min-fold | proxy |
|---|---|---|---|---|
| base (no LM) | 0.4874 | 0.5151 | 0.4529 | 0.4788 |
| ESM-2 650M alone (CPU) | 0.5224 | 0.5521 | 0.4945 | — |
| AntiBERTa2 202M alone (CPU) | 0.5269 | 0.5586 | 0.4885 | — |
| IgBert 420M | 0.5445 | 0.5770 | 0.5046 | 0.5345 |
| AntiBERTa2 + IgBert | 0.5552 | 0.5893 | 0.5086 | 0.5436 |
| **AntiBERTa2 + IgBert + ESM-2** | **0.5650** | **0.5979** | **0.5234** | **0.5546** |

Each encoder adds on top of the previous one and min-fold rises with it (0.4529 -> 0.5234),
which is the half of the metric that punishes collapsing on the harder hidden family.

**Memorisation audit — passed.** The contract flagged an implausible jump from a
paired-pretrained checkpoint as an audit trigger. Discriminator: ESM-2 650M is trained on
UniRef50 general proteins and has never seen an antibody heavy/light pairing, so it cannot
have memorised any. Its fold-0 score is 0.5533 against base 0.5183 — it captures most of the
lift on its own. The ordering IgBert 0.5700 > AntiBERTa2 0.5599 > ESM-2 0.5533 > base 0.5183 is
a smooth gradient in antibody-domain specificity, not the discontinuity memorisation would
produce, and the gain is uniform across all five held-out-donor folds. Verdict: the LM lift is
generic protein sequence representation, and the embeddings are promoted.

## Decisive findings so far
1. **SHM concordance is the dominant mechanism.** Per-V-gene positional germline consensus in
   *CDR3-anchored backwards coordinates* (robust to per-study 5' trimming) gives a heavy/light
   mutation-rate correlation of **0.30–0.42 within block**. Alone, matching the within-block
   *rank* of heavy mutation load to the candidates' rank scores **adjusted 0.357** in-sample
   (w=110). Rank-matching beats absolute-difference matching (0.357 vs 0.326) — scale/offset
   differ per sample, ranks do not. Confirmed by literature (Heavy2Light: "maturation state
   concordance drives pairing compatibility").
2. **Germline pairing bias is real but second.** heavy_V x light_V shrunk log-lift: adjusted
   0.262 in-sample (needs CV — 253x252 cells, overfit risk). heavy_V -> kappa/lambda locus
   preference spans 0.48..0.69, adjusted 0.037. heavy_J x light_J 0.021. V-family x V-family 0.072.
3. **CDR3 length/charge complementarity is ~nil**: within-sample corr +0.038 (length),
   -0.046 (charge). Not a useful axis on its own.
4. **Clonal relatives inside a block are rare but decisive when present**: 17/42000 within-block
   pairs have heavy-CDR3 identity >=0.7 with the same V gene; those pairs share light V 100% of
   the time and have light-CDR3 identity 0.79 (vs 0.044 / 0.124 for other pairs). ~1% of blocks.
5. Naive in-sample combo of (locus + hVxlV + hJxlJ + SHM) = adjusted 0.351.

## Validation
`sample_id` is the private group boundary (preflight: 105 keys, 0.0% test overlap). Test samples
are disjoint from train samples. Two hidden families (held_out_subject / cross_study) are not
labelled. Plan: GroupKFold on `sample_id` = held_out_subject proxy; leave-one-fingerprint-cluster-out
= cross_study proxy. Resolution: 4096 test rows, 1 row = 0.024% of the per-row metric; `adjusted`
doubles per-row deltas, so 1 row ~= 0.0007 adjusted. Selection deltas below ~0.005 adjusted on a
30k-row CV are not resolvable.

## Contract: techniques excluded, with the sentence that excludes each
| Excluded | Sentence |
|---|---|
| Looking a heavy chain up in `train.csv` to retrieve its partner | "Heavy-chain amino-acid sequences are globally deduplicated, so no heavy chain appears twice anywhere in the challenge and none of them can be looked up in `train.csv`." |
| Submitting fewer than 8 indices / partial rankings | "Submitting fewer than 8 indices is always worse in expectation than submitting a full ranking, so rank all 8." |
| Any clock/throughput/OOM/device-triggered branch that changes the training or inference plan | Platform **Deterministic Execution** gate (not `inst.md`) — see `eris-no-wallclock-safeguard` |
| `torch.cuda.is_available()` fallback; device chosen at runtime | Same gate; `inst.md` "Resource Limits: GPU: 1x NVIDIA A10G" fixes the device |

**Stated to be uninformative rather than banned** (so: not worth building, but not a rule
violation if a model incidentally sees them): donor/study/species-level features *within* a block,
candidate slot order, `id`/`block_id` numbering, row order.

**Explicitly NOT excluded — no sentence bans these, and I will use them:**
- Joint use of all 8 heavy chains of a block at inference. `inst.md` states the bijection as a
  fact of the task ("the mapping is a **bijection**") and ships every block whole inside `test.csv`.
  Sinkhorn / assignment decoding is therefore in-contract. Blocks are self-contained, so this is
  still block-local, not transductive across the test set.
- Fitting germline consensus tables and gene-pair lift tables on `train.csv` as model parameters.
- Training on the mouse/rat rows as well as human.
- A general pretrained protein/antibody language model fine-tuned on the supplied train data,
  **including checkpoints pretrained on paired OAS repertoires** (IgBert/IgBert-paired, AbLang2,
  AntiBERTa2). I first restricted myself to unpaired-pretrained checkpoints out of a leakage
  worry; a contract audit correctly found no sentence supporting it, and `inst.md` line 123
  discusses pretrained checkpoints explicitly *without* barring any ("there is no widely
  available off-the-shelf tool or pretrained checkpoint that performs this block deconvolution").
  Restriction dropped. A released checkpoint is a pretrained prior, not external task data; the
  skill's ban is on adding external *datasets* to training. Residual risk retained as an audit
  trigger: if a paired-pretrained LM produces an implausible jump, check for memorisation before
  promoting it.

### Also in the contract (from the audit, hard rules whose violation is fatal)
- Submission must have exactly the columns `id,ranking`, every test `id` exactly once, no
  duplicates, no extra columns — "rejected outright" otherwise.
- `python3 solution.py <public_dataset_dir> <submission_csv>`; two positional arguments.
- One A10G, no multi-GPU. This is the only compute ceiling `inst.md` states.
- The two hidden families are **never identified in `test.csv`** — no per-family calibration,
  routing or family-conditional decoding is possible at inference.
- Train/test are disjoint at source-group level, so no per-donor/per-sample parameter can be
  carried from train to test. Every fitted table must be keyed on things that recur across donors
  (genes, sequence statistics), never on `sample_id`.
- "There is no lookup table or fixed template that recovers the pairing: it has to be modelled."
  This denies a *fixed* table, not fitted parameters — the learned lift/consensus tables are fine.
- A constant ranking, a random ranking, and ranking by position all score exactly 0.

### Region pooling (GPU job `abpair-pool1`)
| pooling | OOF | human | min-fold |
|---|---|---|---|
| whole chain (mean) | 0.5657 | 0.5984 | 0.5248 |
| **mean + CDR3 span** | **0.5739** | **0.6067** | **0.5345** |
| mean + CDR3 + FR3 | 0.5749 | 0.6061 | 0.5298 |

Adding FR3 moves OOF by +0.001 and min-fold by -0.005 — below what this split resolves — while
tripling the embedding width, so `mean + CDR3` is the recipe.

## What the min(family) term is actually worth
The 5-fold min-fold proxy (0.5345) is **misleading, and pessimistically so**. Re-aggregating the
OOF predictions over coherent sample clusters showed clusters scoring 0.03-0.09, which turned
out to be *entirely* the mouse and rat samples: human 0.5979, mouse_BALB/c 0.0714,
mouse_C57BL/6 0.0569, rat_SD 0.0521. **The test set is 100% human**, so those never enter a test
family. Simulating a family as 7 random whole human samples (~2200 rows, the right size) gives
mean 0.6011, p5 0.5459, min 0.5156.

**Honest expected platform score ≈ 0.585** = 0.75 * 0.5979 + 0.25 * 0.546.

## Delivery
`solution.py` (sha256 `9c31e43aa0ea96d5...`), cold-run as `abpair-cold2`. An earlier cold run
(`abpair-cold1`, mean-pooling only) completed in **14.4 min on an RTX A5000**, exit 0, 4096 valid
rows — so the pipeline is proven end to end and the delivered plan sits at roughly 16-24% of a
90-minute budget even allowing an A10G to be ~1.5x slower.

## Maximize pass (second invocation)

Diagnosis first: the incumbent's decoded errors show **no structural defect**. Top-1 is 42.9%
against a 12.5% chance rate, the rank histogram decays smoothly, top-1 marginal calibration is
monotone and well ordered (conf 0.26 -> acc 0.21, conf 0.95 -> acc 0.86), there is no
kappa/lambda asymmetry (0.603 / 0.612) and no subgroup collapse. That rules out decoder,
preprocessing and validation faults and points at representation and capacity.

### Capacity / budget sweep (`abpair-sweep1`, 5 folds, control = 3 seeds, OOF 0.5768 / human 0.6079)
| arm | OOF | human | verdict |
|---|---|---|---|
| seeds 1 | 0.5584 | 0.5884 | — |
| seeds 8 | 0.5826 | 0.6150 | **+0.0071 human** |
| seeds 16 | 0.5852 | 0.6168 | **+0.0089 human** |
| **embedding dropout 0.45** | 0.5841 | 0.6149 | **+0.0070 human at only 3 seeds** |
| human-only training | 0.5765 | 0.6077 | **dead heat** — mouse/rat rows neither help nor hurt |
| per-seed marginal averaging | 0.5820 | 0.6140 | slightly worse than logit averaging |
| projection width 128 | 0.5706 | 0.6025 | rejected |
| hidden 320 | 0.5741 | 0.6061 | neutral |
| 22 epochs | 0.5651 | 0.5975 | rejected, overfits |
| column loss 1.0 / 0.0 | 0.5743 / 0.5762 | 0.6065 / 0.6087 | neutral, keep 0.5 |
| pair dropout 0.25 | 0.5760 | 0.6081 | neutral |

Only two knobs move anything, and they are different mechanisms: **ensembling** and
**regularising the embedding towers**. Everything else is inside the noise. Note the delivered
8-seed recipe was therefore already worth ~0.583 OOF / 0.615 human, not the 0.5739 / 0.6067 I
had quoted from the 3-seed sweep.

### Fine-tuning, retested fairly (`abpair-ft2`)
The first probe was not a fair test — it compared a sequence-only cross-encoder against
frozen-LM-plus-features. Retested properly: AntiBERTa2 fine-tuned per fold on that fold's
training blocks only, its pooled embeddings substituted for the frozen ones inside the *same*
full model. The all-frozen control reproduced the incumbent exactly (fold 0: 0.6026 vs 0.6026),
so the comparison is clean. **Fine-tuned: 0.5968, i.e. -0.0058.** Training loss fell 2.71 ->
2.35 -> 1.90, the signature of a 202M encoder memorising ~3000 blocks and losing the general
representation that transfers across the donor boundary. Consistent with ImmunoMatch's published
drop from 0.75 in-distribution to 0.66 on external donors. Stopped after one fold on low
decision value: the mechanism needed a large win to justify ~30 min of evaluator runtime, and a
negative first fold rules that out. Not refuted — one fold — but not worth further spend.

### Confirming the combination (`abpair-conf1`, 5 folds, 16 seeds)
| arm | OOF | human |
|---|---|---|
| edrop 0.30 | 0.5852 | 0.6168 |
| **edrop 0.45** | **0.5904** | **0.6221** |
| edrop 0.55 | 0.5846 | 0.6154 |
| edrop 0.45 + pdrop 0.25 | 0.5912 | 0.6223 |

The two wins compose (+0.0053 over edrop 0.30 at the same seeds) and embedding dropout has a
sharp interior optimum at 0.45. Adding pair dropout on top moves OOF by +0.0008 and human by
+0.0002 — below what this split resolves — so it is not taken; fewer changes from the verified
path. The edrop-0.30 arm reproduced sweep1 exactly (0.5852 / 0.6168 in both jobs).

**Promoted to delivery: 16 seeds, embedding dropout 0.45. OOF 0.5904 / human 0.6221.**

### Re-blocking augmentation (`abpair-rb1`) — CONFOUNDED, not a clean negative
The 3797 shipped blocks are one arbitrary partition of each sample's 320 cells out of ~2.5e15
valid ones. Re-partitioning within sample was verified structurally sound (bijection in every
block, balanced targets, true partner preserved, slots alphabetical, 3% of rows dropped where a
block would have held two identical light chains). Measured: K=2 extra partitions scored 0.5375
against a 0.5719 control, i.e. **-0.0364**.

That test does not mean what it looks like. Tripling the data at a fixed 14 epochs triples the
optimizer updates, and the sweep already showed 22 epochs overfitting (0.5651 vs 0.5768). The
arm compared *3x the label exposure*, not re-blocking. A fair test holds updates constant
(K=2 at ~5 epochs). Job stopped rather than burn 45 more minutes on two more confounded arms.
Recorded as unresolved, not refuted.

## Fair negatives (do not retry without a new reason)
- Probabilistic surprisal SHM clock (0.311 vs 0.364 for plain hamming).
- Two-pass germline consensus (identical to one-pass at every keep fraction).
- Two-anchor FR2 alignment (0.342 vs 0.364) — CDR1/CDR2 lengths are germline-constant.
- Per-position residue towers (train 0.76 / val 0.45).
- End-to-end fine-tuning of IgBert (fold-0 0.489 sequence-only vs 0.570 frozen+features after
  one epoch, 10 min/epoch) — stopped, not disproven; see below.

## Largest unresolved weaknesses / remaining opportunity
1. **Unused runtime budget.** The delivered plan uses ~16-24% of the 90 minutes. The obvious
   spend is more seeds (8 -> 20, about +3 min) and it was skipped only because the deadline
   arrived, not because it was measured and rejected.
2. **Fine-tuning was stopped, not refuted.** The probe trained a *sequence-only* cross-encoder
   with no access to the SHM clock or lift tables, so 0.489-vs-0.570 is not a like-for-like
   comparison. The right test is to fine-tune the encoder and then feed its embeddings into the
   full model. At 10 min/epoch it is affordable in the evaluator budget.
3. **Mouse/rat rows (6% of train) score ~0.06** and are plausibly injecting gradient noise into
   a model that is only ever scored on human. Training human-only is free at delivery and
   untested; recorded as a decision made without evidence.
4. Head hyperparameters (projection width, embedding dropout, epochs) were never swept — that
   arm of `abpair-pool1` was cut for time.

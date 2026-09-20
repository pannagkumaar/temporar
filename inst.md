ChallengeEvaluation

[Antibody Light-Chain Pairing Reconstruction](/quests/eris/challenges/jx71hsctdsj6z6av58xykyztqn8e7s27)
-------------------------------------------------------------------------------------------------------

OthermediumAccepted

Higher score is better

Score range: 0 to 1

#### Problem Description

**Antibody Chain Pairing Deconvolution**

**Overview**

An antibody is built from two different protein chains — a **heavy chain** and a **light chain** — that fold together to form the antigen-binding site. Each B cell makes its own private combination of the two.

In plain terms: **you are given 8 heavy chains and the 8 light chains that belong to them, but the pairing has been scrambled. Work out which light chain goes with which heavy chain.**

This is a real problem, not a puzzle invented for this challenge. Antibody repertoires are usually sequenced in *bulk*, which reads heavy chains and light chains out of separate pools of cells and **destroys the pairing information**. Only expensive single-cell sequencing preserves it. Being able to computationally re-pair chains is directly useful for antibody discovery and engineering pipelines that have to work from bulk data.

It is also genuinely hard. The two chains are produced by independent semi-random DNA-shuffling events (V(D)J recombination), so no formula derives one from the other. But they are not independent either — the two chains of one cell share a history: they must fold together into a stable protein, they went through the same rounds of affinity maturation (so they carry correlated amounts of somatic mutation), and certain heavy germline genes pair with certain light germline genes more often than chance. Those weak, statistical, real signals are what a good solution has to find.

Every sequence here is real: each row is one real human, mouse or rat B cell captured by single-cell antibody repertoire sequencing in a published immunology study, with both of its chains recovered and annotated.

**Task**

Both `train.csv` and `test.csv` are organised into **blocks**. Each block is 8 real B cells drawn from **one single real sample** (one donor, one sequencing run), with their pairings scrambled. Every row is one heavy chain, and each row carries its own block's 8 candidate light chains in the columns `cand1_code`…`cand8_code` and `cand1_aa`…`cand8_aa`. The 8 rows sharing a `block_id` carry the identical candidate list.

Exactly one candidate light chain is the true partner of each heavy chain, and the mapping is a **bijection**: every candidate is the partner of exactly one heavy chain in that block. Because the decoys come from the same sample as the true partner, they share the same donor, the same sequencing run, and the same repertoire statistics — you cannot tell them apart with sample-level features.

`train.csv` tells you the answer for each of its rows in the `true_candidate_index` column (which of that row's 8 candidates is the real partner, 1–8). `test.csv` has exactly the same columns except that one.

For each test heavy chain, submit a **ranking of that block's 8 candidates**, best guess first. You are scored on how highly you rank the true partner, so a confident-but-wrong answer that still places the truth second costs you much less than one that buries it last.

**Dataset Files**

`public/train.csv` (30,376 rows in 3,797 blocks) and `public/test.csv` (4,096 rows in 512 blocks) have **identical columns**, except that `train.csv` additionally carries the target column `true_candidate_index`. One row = one heavy chain.

* `id` — string — Unique row id (`tr_000001` in train, `ab_000001` in test). Test ids are what you submit against.
* `block_id` — string — Which block this heavy chain belongs to. Exactly 8 rows share each `block_id`, and they all list the same 8 candidates.
* `sample_id` — string — Anonymised sample label, e.g. `smp_017`. Rows sharing a `sample_id` came from the same donor/sequencing run; several blocks can come from one sample.
* `species` — string — `human`, `mouse_BALB/c`, `mouse_C57BL/6`, or `rat_SD`.
* `heavy_v_gene` — string — Heavy-chain germline V gene, allele suffix stripped, e.g. `IGHV3-23`.
* `heavy_d_gene` — string — Heavy-chain germline D gene, e.g. `IGHD3-10` (may be empty when not confidently called).
* `heavy_j_gene` — string — Heavy-chain germline J gene, e.g. `IGHJ4`.
* `heavy_cdr3_aa` — string — Heavy-chain CDR3 amino-acid sequence.
* `heavy_chain_aa` — string — Full heavy-chain variable-region amino-acid sequence (contains `heavy_cdr3_aa` as a substring).
* `cand1_code` … `cand8_code` — string — The block's 8 candidate light chains, each packed as `LIGHT_V_GENE|LIGHT_J_GENE|LIGHT_CDR3_AA`, e.g. `IGKV1-39|IGKJ1|QQSYSTPLT`. Split on the `|` character for the three fields.
* `cand1_aa` … `cand8_aa` — string — The same 8 candidates' full light-chain variable-region amino-acid sequences. `candN_aa` is the sequence of the candidate whose code is `candN_code`.
* `true_candidate_index` — integer — `train.csv` **only.** Which candidate, 1–8, is this heavy chain's real partner.

The candidate slots are ordered alphabetically by `light_chain_code`, which depends only on the candidate itself and carries no information about which heavy chain it belongs to.

All amino-acid sequences use only the 20 canonical letters (`ACDEFGHIKLMNPQRSTVWY`).

**Submission Format**

Submit a CSV with exactly two columns and one row per test `id`:

```
id,ranking
```

`ab_000001,7;3;1;8;2;5;4;6`

`ab_000002,2;4;6;1;3;7;8;5`

* `ranking` is that block's `candidate_index` values, **best guess first**.
* Canonical separator is `;`. Commas and whitespace are also accepted.
* A full ranking is 8 distinct indices in the range 1–8.
* Every test `id` must appear exactly once. No extra columns.

**Malformed and partial rankings.** Out-of-range values, repeats after the first occurrence, and trailing junk are discarded rather than rejected. If the true partner does not appear anywhere in what survives parsing (an empty, unparseable or truncated ranking), the row is scored as if you had ranked the true partner **last**. Nothing about a bad row raises an error — it just earns no credit. Submitting fewer than 8 indices is always worse in expectation than submitting a full ranking, so rank all 8.

A submission with the wrong columns, a missing test `id`, or duplicate ids is rejected outright.

`sample_submission.csv` ranks `1;2;3;4;5;6;7;8` for every row — a valid, zero-information submission that scores exactly 0.

**Evaluation**

For one test row, let `n` be the block size (8) and let `rank` be the position of the true partner in your submitted ranking (1 = first):

```
row_credit = (n - rank) / (n - 1)
```

so ranking the true partner first earns 1.0, last earns 0.0, and a uniformly random ranking earns 0.5 on average. Chance is then rescaled away, so that guessing is worth nothing:

```
adjusted(rows) = clip(2 * mean(row_credit over rows) - 1, 0, 1)
```

The test set contains two hidden families, **never identified in** `test.csv`:

* `held_out_subject` — cells from a donor/sample that contributes no rows to `train.csv`, but from a study that does contribute other donors.
* `cross_study` — cells from a study that contributes **no** rows to `train.csv` at all: fully out-of-distribution generalization.

The final score rewards overall ranking quality while requiring that a solution not collapse on whichever family is harder for it:

```
score =
```

`` `0.75 * adjusted(all test rows)` ``

`+ 0.25 * min(adjusted(held_out_subject rows), adjusted(cross_study rows))`

Score is clipped to `[0, 1]`. Ranking every true partner first scores 1.0. Any strategy that carries no information about the true pairing — a constant ranking, a random ranking, ranking by position — scores 0.

**Computing your own score.** `train.csv` is already built exactly like `test.csv` — same blocks of 8, same candidate slots, same alphabetical ordering — and it comes with `true_candidate_index`, so you can score yourself locally with the formula above directly. Holding out whole `sample_id` values for validation will approximate the real test conditions much better than a random row split, since the test blocks come from donors and studies that contribute no training rows at all.

**What Is and Isn't a Shortcut**

The train/test split is disjoint at the level of real-world **source group** (study × subject, or study × sample where subject identity was not tracked): no donor or sample contributing to `test.csv` contributes any row to `train.csv`. Heavy-chain amino-acid sequences are globally deduplicated, so no heavy chain appears twice anywhere in the challenge and none of them can be looked up in `train.csv`.

Within a block, decoys come from the same sample as the true partner, so donor-level, study-level and species-level features do not discriminate. Candidate order is alphabetical by candidate content and independent of the heavy chains; row order, `id` values and `block_id` numbering are assigned after content-determined sorting and hashing, and carry no information about the answer or about which family a block belongs to. There is no lookup table or fixed template that recovers the pairing: it has to be modelled.

**Nearest Prior Art**

Tools such as IGoR and OLGA model V(D)J recombination statistics for **one chain in isolation**, and codon-optimization models such as CodonTransformer solve an unrelated protein-to-DNA problem; neither addresses cross-chain pairing. Heavy/light pairing prediction is an open, only-partially-solved problem in antibody informatics, and there is no widely available off-the-shelf tool or pretrained checkpoint that performs this block deconvolution against real single-cell ground truth with a source-disjoint, two-family evaluation split.

**Resource Limits**

* GPU: 1x NVIDIA A10G

Expected Output

Your script receives the public dataset directory and exact submission CSV path as two positional arguments.
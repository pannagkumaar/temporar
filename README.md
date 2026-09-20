# Antibody Light-Chain Pairing Reconstruction

Reconstructing which light chain belongs to which heavy chain inside blocks of 8 real
single-cell-sequenced B cells, where the pairing has been scrambled.

`python3 solution.py <public_dataset_directory> <submission_csv_path>`

## Result

| | 5-fold sample-grouped CV |
|---|---|
| OOF adjusted (all rows) | **0.5739** |
| OOF adjusted (human rows only — the test set is 100% human) | **0.6067** |
| Chance / `sample_submission.csv` | 0.0 |

Expected platform score ≈ **0.585**, from `0.75 * 0.5979 + 0.25 * 0.546`, where the second
term is the 5th percentile of a simulated 7-sample held-out family.

## Mechanism

Two cells' chains are produced by independent V(D)J recombination, so neither derives from the
other. But the two chains of *one* cell share a history, and three signals carry it:

1. **Somatic-hypermutation concordance** (dominant). Both chains went through the same rounds of
   affinity maturation, so their mutation loads correlate — measured at **0.41 within block**.
   The clock is a hamming distance to a per-V-gene positional consensus fitted on the training
   data in **CDR3-anchored backwards coordinates**, which makes it immune to the per-study
   differences in 5′ trimming. It is compared as a *within-block rank*, because the absolute
   scale differs per donor (rank matching 0.364 vs absolute difference 0.326, alone).
2. **Germline pairing preference**. Shrunk log-lift tables over heavy/light gene pairs at
   several granularities. Training rows are scored with their own observation removed from the
   table (exact leave-one-out) — worth +0.0085, because otherwise the feature leaks its own
   label and the model over-trusts a feature that is weaker at inference.
3. **Pretrained sequence representation**. Frozen embeddings from three encoders concatenated
   (AntiBERTa2 202M, IgBert 420M, ESM-2 650M), pooled twice per chain — whole chain and CDR3
   span — entering through learned projections and a bilinear interaction.

## Decoding

The block is a bijection, so it is a permutation model over pairwise log-potentials. The metric
pays `(8-rank)/7`, which is maximised by ranking candidates by their **true marginal posterior**
`P(pi(i)=j)`. With n=8 that marginal is computed *exactly* via Ryser's formula (64 permanents of
7×7 minors per block, 2.2 s for 3797 blocks) instead of being approximated by Sinkhorn.

## What moved the score

| step | OOF adj |
|---|---|
| v1 features, Sinkhorn decode, 1 seed | 0.4628 |
| probabilistic-surprisal SHM clock | 0.4441 — **rejected** |
| per-position residue towers | ~0.43 — **rejected** (train 0.76 / val 0.45) |
| two-anchor germline alignment | 0.342 single-feature vs 0.364 — **rejected** |
| v3 features + exact leave-one-out lift tables | 0.4758 |
| 3 seeds | 0.4794 |
| exact permutation marginals (vs Sinkhorn) | 0.4850 |
| + IgBert frozen embeddings | 0.5445 |
| + AntiBERTa2 | 0.5552 |
| + ESM-2 | 0.5650 |
| + CDR3-span pooling | **0.5739** |

## Notes

* `working/solution_notes.md` — full scientific record, contract and fair negatives.
* `working/research/` — research code (CV harness, feature builders, decoders, audits).
* The model scores ~0.06 on the mouse/rat training rows and 0.598 on human. The test set is
  entirely human, so this does not affect the delivered score, but it means ~6% of the
  training rows contribute almost nothing.
* `solution.py` downloads three checkpoints from the Hugging Face Hub at run time and requires
  `transformers` and a CUDA device. AntiBERTa2 is read with `BertTokenizer` rather than its
  `AutoTokenizer` to avoid an `rjieba` dependency; the substitution was verified to produce
  byte-identical `input_ids` on 800 probe sequences.

# Antibody Light-Chain Pairing Reconstruction

Reconstructing which light chain belongs to which heavy chain inside blocks of 8 real
single-cell-sequenced B cells, where the pairing has been scrambled.

`python3 solution.py <public_dataset_directory> <submission_csv_path>`

## Result

| | 5-fold sample-grouped CV |
|---|---|
| OOF adjusted (all rows) | **0.5904** |
| OOF adjusted (human rows only — the test set is 100% human) | **0.6221** |
| Top-1 accuracy | 42.9% (chance 12.5%) |
| `sample_submission.csv` / chance | 0.0 |

Cold-run verified: RTX A6000, exit 0, 20.0 min, 4096 valid rows.

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
   span.

## Decoding

The block is a bijection, so it is a permutation model over pairwise log-potentials. The metric
pays `(8-rank)/7`, which is maximised by ranking candidates by their **true marginal posterior**
`P(π(i)=j)`. With n=8 that marginal is computed *exactly* via Ryser's formula (64 permanents of
7×7 minors per block, 2.2 s for 3797 blocks) instead of being approximated by Sinkhorn.

## What moved the score

| step | OOF adj |
|---|---|
| v1 features, Sinkhorn decode, 1 seed | 0.4628 |
| v3 features + exact leave-one-out lift tables | 0.4758 |
| exact permutation marginals (vs Sinkhorn) | 0.4850 |
| + IgBert frozen embeddings | 0.5445 |
| + AntiBERTa2 | 0.5552 |
| + ESM-2 | 0.5650 |
| + CDR3-span pooling | 0.5739 |
| + 16 seeds | 0.5852 |
| + embedding dropout 0.45 | **0.5904** |

## Fair negatives (measured, not assumed)

| tried | result |
|---|---|
| Fine-tuning AntiBERTa2 end-to-end, embeddings substituted into the same model | 0.5968 vs 0.6026 frozen on fold 0; train loss 2.71→1.90, a 202M encoder memorising ~3000 blocks |
| Probabilistic-surprisal SHM clock | 0.311 vs 0.364 — tracks germline allelic diversity, not mutation |
| Two-anchor germline alignment on the conserved FR2 tryptophan | 0.342 vs 0.364 — CDR1/CDR2 lengths are germline-constant, so there was no drift to fix |
| Two-pass consensus from the least-mutated half | identical at every keep fraction |
| Per-position residue towers (VH/VL interface) | train 0.76 / val 0.45 |
| Human-only training (mouse/rat rows score 0.06) | dead heat, 0.6077 vs 0.6079 |
| Blending structurally diverse configs instead of more seeds | 0.5850 vs 0.5852 — ensembling has saturated |
| Within-sample re-blocking augmentation (the 40 shipped blocks per sample are one partition out of ~10¹⁵) | 0.5320 vs 0.5861 **at matched optimizer updates** — the shipped partition is evidently not uniformly random |
| A 4th encoder - AntiBERTa2-CSSP (structure-contrastive, the one with a mechanism) | 0.6188 vs 0.6211 control; encoder stacking saturates at 3 |
| A 4th encoder - IgBert_unpaired (near-null control) | 0.6200 vs 0.6211 |
| Residue-level cross-chain interaction (24 CDR3-anchored frozen-LM residue vectors per chain, learned 24x24 interaction weights) | 0.6178 vs 0.6211, losing on 4/5 folds - pooled-vector interaction is not leaving reachable residue information behind |
| Block-level SHM-dispersion gate features | no effect (+/-0.001, below resolution) in two independent runs |
| Marginal temperature, global sweep and block-adaptive | +0.0003 / +0.0005 - noise, with non-monotone fitted temperatures |
| Projection width 128 / hidden 320 / 22 epochs / column-loss 0 or 1 / pair dropout 0.25 | all neutral or worse |

## Notes

* `working/solution_notes.md` — full scientific record and contract; `working/experiments.jsonl`
  — the run ledger; `working/research/` — CV harness, feature builders, decoders, audits.
* **Memorisation audit.** A paired-pretrained antibody LM lifting the score raises a leakage
  worry. The discriminator is a model that *cannot* have memorised: ESM-2 is trained on general
  proteins and has never seen a VH/VL pairing, and it captured most of the lift on its own
  (0.5224 alone). The ordering IgBert > AntiBERTa2 > ESM-2 > none is a smooth gradient in
  antibody-domain specificity, not the discontinuity memorisation would produce.
* **The min-over-families term.** A naive min-over-folds proxy read 0.5345, but that was
  contaminated: the near-zero clusters are entirely mouse/rat (0.05–0.07) while human is 0.598.
  The test set is 100% human, so the honest family risk is ~0.546 (5th percentile of simulated
  7-sample human families).
* `solution.py` downloads three checkpoints from the Hugging Face Hub at run time and requires
  `transformers` and a CUDA device. AntiBERTa2 is read with `BertTokenizer` rather than its
  `AutoTokenizer` to avoid an `rjieba` dependency; the substitution was verified to produce
  byte-identical `input_ids` on 800 probe sequences.

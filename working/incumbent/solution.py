#!/usr/bin/env python3
"""Antibody Light-Chain Pairing Reconstruction.

    python3 solution.py <public_dataset_directory> <submission_csv_path>

Mechanism
---------
Two cells' chains are not derivable from each other, but the two chains of one cell share a
history. Three signals carry that, in descending order of measured strength:

1. Somatic-hypermutation concordance. Both chains of a cell went through the same rounds of
   affinity maturation, so their mutation loads correlate (0.41 within block, measured). The
   clock is a hamming distance to a per-V-gene positional consensus built from the training
   data in *CDR3-anchored backwards coordinates*, which makes it immune to the per-study
   differences in 5' sequence trimming. It is compared as a within-block rank, not an absolute
   value, because the scale differs by donor.
2. Germline pairing preference. Shrunk log-lift tables over heavy/light gene pairs at several
   granularities (gene, family, locus, J, CDR3-length bucket). Training rows are scored with
   their own observation removed from the table (exact leave-one-out); without that the feature
   leaks its own label and the model over-trusts it.
3. Pretrained sequence representation. Frozen embeddings of both chains from three encoders
   concatenated (AntiBERTa2 202M, IgBert 420M, ESM-2 650M), pooled twice per chain (whole chain
   and CDR3 span), entering through learned projections and a bilinear interaction. Each
   encoder added score on the same folds: none 0.4874 -> IgBert 0.5445 -> +AntiBERTa2 0.5552
   -> +ESM-2 0.5650, and CDR3-span pooling -> 0.5739 (OOF adjusted, 3 seeds).

The encoders are used frozen. Fine-tuning AntiBERTa2 per fold with this same listwise loss and
substituting its embeddings scores 0.5968 against 0.6026 for frozen on fold 0: a 202M encoder
memorises ~3000 blocks (training loss 2.71 -> 1.90) and loses the general representation that
transfers across the donor boundary.

Head settings are the measured optimum, not defaults. Embedding dropout peaks sharply at 0.45
(0.30 -> 0.5852, 0.45 -> 0.5904, 0.55 -> 0.5846 OOF at 16 seeds) because the 6656-dimensional
embedding towers are where this model overfits. Seed averaging pays to ~16 and then saturates
(1 -> 0.5584, 3 -> 0.5768, 8 -> 0.5826, 16 -> 0.5852); blending structurally diverse
configurations instead of seeds gives nothing extra (0.5850).
Final: **OOF adjusted 0.5904, human-only 0.6221.**

Decoding
--------
The block is a bijection, so it is a permutation model over pairwise log-potentials. The metric
pays (8-rank)/7, which is maximised by ranking candidates by their true marginal posterior.
With n=8 that marginal is computed exactly via Ryser's formula rather than approximated by
Sinkhorn (worth +0.006 over the Sinkhorn output the model was trained with).

The plan is fixed: constant seeds, epochs, batch size and model count, no runtime branching.
"""
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel, BertTokenizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(8)

DEVICE = torch.device('cuda')          # inst.md fixes the hardware: 1x NVIDIA A10G
# Three pretrained encoders, concatenated. Each one added measurable score on the same folds:
# none 0.4874 -> IgBert 0.5445 -> +AntiBERTa2 0.5552 -> +ESM-2 0.5650 (OOF adjusted).
# The tokenizer class is pinned per repo rather than chosen at runtime: AntiBERTa2 is a RoFormer
# whose AutoTokenizer pulls in rjieba (a Chinese segmenter it does not need for a 20-letter
# alphabet), so it is read with BertTokenizer over the same vocab.txt, which is verified to
# produce identical input_ids.
LM_SPECS = (('alchemab/antiberta2', BertTokenizer),
            ('Exscientia/IgBert', AutoTokenizer),
            ('facebook/esm2_t33_650M_UR50D', AutoTokenizer))
LM_MAXLEN = 160
LM_BATCH = 64
N_SEEDS = 16
EPOCHS = 14
BLOCKS_PER_BATCH = 96
LR = 3e-3
WD = 1e-4
HID = 192
DGENE = 48
DEMB = 64
PDROP = 0.15
EDROP = 0.45
COLW = 0.5
SINK_TRAIN = 6
MARGINAL_T = 1.25

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}
GAP = 20
SHM_WINDOWS = ((0, 110), (0, 40), (40, 75), (75, 110))
LIFT_SPECS = (('vv', 30), ('vfam', 20), ('famv', 20), ('famfam', 10), ('vloc', 20),
              ('famloc', 10), ('jj', 20), ('jv', 30), ('vj', 30), ('jloc', 10),
              ('dv', 30), ('lenv', 30), ('lenloc', 10))
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6,
                   -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CHG = {c: (1.0 if c in 'KR' else (-1.0 if c in 'DE' else (0.5 if c == 'H' else 0.0))) for c in AA}


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- data
def load_split(path, with_target):
    df = pd.read_csv(path)
    n = len(df)
    codes = df[[f'cand{i}_code' for i in range(1, 9)]].values.astype(object)
    aas = df[[f'cand{i}_aa' for i in range(1, 9)]].values.astype(object)
    lv = np.empty((n, 8), dtype=object)
    lj = np.empty((n, 8), dtype=object)
    lc = np.empty((n, 8), dtype=object)
    for i in range(n):
        for j in range(8):
            a, b, c = codes[i, j].split('|')
            lv[i, j] = a
            lj[i, j] = b
            lc[i, j] = c
    df['heavy_d_gene'] = df['heavy_d_gene'].fillna('NA')
    y = (df['true_candidate_index'].values - 1) if with_target else None
    return dict(df=df, aas=aas, lv=lv, lj=lj, lc=lc, y=y)


def block_order(block_ids):
    """Index array grouping rows block-major, 8 per block. Sorted, so it is hash-stable."""
    s = pd.Series(np.arange(len(block_ids)), index=block_ids)
    order = np.concatenate([g.values for _, g in s.groupby(level=0, sort=True)])
    return order, len(order) // 8


# ------------------------------------------------------- germline consensus / SHM clock
def encode_back(seqs, cdr3s, w):
    out = np.full((len(seqs), w), GAP, dtype=np.int8)
    for i in range(len(seqs)):
        s = seqs[i]
        p = s.rfind(cdr3s[i])
        if p > 0:
            k = min(w, p)
            out[i, :k] = [AA2I.get(s[p - 1 - j], GAP) for j in range(k)]
    return out


class BackConsensus:
    """Per-V-gene modal residue at each CDR3-anchored backwards position."""

    def __init__(self, w, min_n=4):
        self.w = w
        self.min_n = min_n

    @staticmethod
    def _cons(arr):
        out = np.full(arr.shape[1], GAP, dtype=np.int8)
        for p in range(arr.shape[1]):
            col = arr[:, p]
            col = col[col != GAP]
            if len(col) >= 3:
                out[p] = np.bincount(col, minlength=21).argmax()
        return out

    def fit(self, genes, seqs, cdr3s):
        enc = encode_back(seqs, cdr3s, self.w)
        self.glob = self._cons(enc)
        buckets = collections.defaultdict(list)
        for i in range(len(genes)):
            buckets[genes[i]].append(i)
        self.cons = {}
        for g in sorted(buckets.keys()):
            idx = buckets[g]
            if len(idx) >= self.min_n:
                self.cons[g] = self._cons(enc[np.array(idx)])
        return self

    def mut(self, genes, seqs, cdr3s):
        enc = encode_back(seqs, cdr3s, self.w)
        n = len(enc)
        rate = np.zeros(n)
        cnt = np.zeros(n)
        cov = np.zeros(n)
        buckets = collections.defaultdict(list)
        for i in range(len(genes)):
            buckets[genes[i]].append(i)
        for g in sorted(buckets.keys()):
            idx = np.array(buckets[g])
            c = self.cons.get(g, self.glob)
            e = enc[idx]
            m = (e != GAP) & (c != GAP)[None, :]
            k = m.sum(1).astype(np.float64)
            d = ((e != c[None, :]) & m).sum(1).astype(np.float64)
            cov[idx] = k
            cnt[idx] = d
            rate[idx] = d / np.maximum(k, 1.0)
        return rate, cnt, cov


class LiftTable:
    """Shrunk log-lift log[P(b|a)/P(b)] with Dirichlet-on-marginal smoothing."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, a, b):
        self.cab = collections.Counter(zip(a, b))
        self.ca = collections.Counter(a)
        self.cb = collections.Counter(b)
        self.N = len(a)
        return self

    def __call__(self, a, b):
        pb = self.cb.get(b, 0) / self.N
        if pb <= 0:
            return 0.0
        num = self.cab.get((a, b), 0) + self.alpha * pb
        den = self.ca.get(a, 0) + self.alpha
        return float(np.log((num / den) / pb))

    def loo(self, a, b_true, b):
        """Same lift with this row's own (a, b_true) observation removed from the counts."""
        N = self.N - 1
        cb = self.cb.get(b, 0) - (1 if b == b_true else 0)
        pb = cb / N
        if pb <= 0:
            return 0.0
        num = self.cab.get((a, b), 0) - (1 if b == b_true else 0) + self.alpha * pb
        den = self.ca.get(a, 0) - 1 + self.alpha
        if den <= 0 or num <= 0:
            return 0.0
        return float(np.log((num / den) / pb))


def seq_props(s):
    n = max(len(s), 1)
    return [len(s),
            sum(KD.get(c, 0.0) for c in s) / n,
            sum(CHG.get(c, 0.0) for c in s),
            sum(c in 'FWY' for c in s) / n,
            s.count('G') / n,
            (s.count('S') + s.count('T')) / n]


def seqsim(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _rank(v, key):
    return pd.Series(v).groupby(pd.Series(key)).rank(method='average').values - 1.0


def _z(v, key):
    return pd.Series(v).groupby(pd.Series(key)).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)).values


def _crank(l):
    return np.argsort(np.argsort(l, 1, kind='stable'), 1).astype(np.float32)


def _cz(l):
    return (l - l.mean(1, keepdims=True)) / (l.std(1, keepdims=True) + 1e-6)


class FeatureBuilder:
    def fit(self, D):
        df = D['df']
        lv, lj, lc, aas, y = D['lv'], D['lj'], D['lc'], D['aas'], D['y']
        r = np.arange(len(df))
        tv, tj, tc, ta = lv[r, y], lj[r, y], lc[r, y], aas[r, y]
        hv = df['heavy_v_gene'].values
        hj = df['heavy_j_gene'].values
        hd = df['heavy_d_gene'].values
        haa = df['heavy_chain_aa'].values
        hc3 = df['heavy_cdr3_aa'].values

        self.hcons = {}
        self.lcons = {}
        for wk in SHM_WINDOWS:
            self.hcons[wk] = BackConsensus(wk[1]).fit(hv, haa, hc3)
            self.lcons[wk] = BackConsensus(wk[1]).fit(tv, ta, tc)
        hr = self.hcons[SHM_WINDOWS[0]].mut(hv, haa, hc3)[0]
        lr = self.lcons[SHM_WINDOWS[0]].mut(tv, ta, tc)[0]
        self.href = {g: np.sort(hr[hv == g]) for g in sorted(set(hv))}
        self.lref = {g: np.sort(lr[tv == g]) for g in sorted(set(tv))}
        self.hall = np.sort(hr)
        self.lall = np.sort(lr)

        hfam = np.array([g.split('-')[0] for g in hv])
        lfam = np.array([g.split('-')[0] for g in tv])
        loc = np.array([g[:3] for g in tv])
        hb = np.clip(df['heavy_cdr3_aa'].str.len().values // 3, 0, 9)
        src = dict(vv=(hv, tv), vfam=(hv, lfam), famv=(hfam, tv), famfam=(hfam, lfam),
                   vloc=(hv, loc), famloc=(hfam, loc), jj=(hj, tj), jv=(hj, tv), vj=(hv, tj),
                   jloc=(hj, loc), dv=(hd, lfam), lenv=(hb, tv), lenloc=(hb, loc))
        self.T = {k: LiftTable(al).fit(*src[k]) for k, al in LIFT_SPECS}
        self.vocab = {}
        for nm, vals in (('hv', hv), ('hj', hj), ('hd', hd), ('lv', tv), ('lj', tj)):
            self.vocab[nm] = {v: i + 1 for i, v in enumerate(sorted(set(vals)))}
        self.vocab['sp'] = {v: i + 1 for i, v in enumerate(sorted(set(df['species'].values)))}
        return self

    @staticmethod
    def _pct_ref(ref_map, allref, genes, vals):
        out = np.empty(len(vals), dtype=np.float32)
        for i in range(len(vals)):
            ref = ref_map.get(genes[i])
            if ref is None or len(ref) < 8:
                ref = allref
            out[i] = np.searchsorted(ref, vals[i], 'left') / max(len(ref), 1)
        return out

    @staticmethod
    def _clonal(blk, hc3, hv, lc, lv):
        n = len(blk)
        f_dot = np.zeros((n, 8), np.float32)
        f_max = np.zeros((n, 8), np.float32)
        order = pd.Series(np.arange(n), index=blk)
        for _, g in order.groupby(level=0, sort=True):
            ii = g.values
            m = len(ii)
            Hs = np.zeros((m, m), np.float32)
            for a in range(m):
                for b in range(a + 1, m):
                    s = seqsim(hc3[ii[a]], hc3[ii[b]]) * (hv[ii[a]] == hv[ii[b]])
                    Hs[a, b] = s
                    Hs[b, a] = s
            j0 = ii[0]
            Ls = np.zeros((8, 8), np.float32)
            for a in range(8):
                for b in range(a + 1, 8):
                    s = seqsim(lc[j0, a], lc[j0, b]) * (lv[j0, a] == lv[j0, b])
                    Ls[a, b] = s
                    Ls[b, a] = s
            hs = -np.sort(-Hs, 1)[:, :3]
            ls = -np.sort(-Ls, 1)[:, :3]
            f_dot[ii] = hs @ ls.T
            f_max[ii] = -np.abs(hs[:, :1] - ls[:, 0][None, :])
        return [f_dot, f_max]

    def transform(self, D, loo):
        df = D['df']
        n = len(df)
        lv, lj, lc, aas = D['lv'], D['lj'], D['lc'], D['aas']
        hv = df['heavy_v_gene'].values
        hj = df['heavy_j_gene'].values
        hd = df['heavy_d_gene'].values
        hc3 = df['heavy_cdr3_aa'].values
        haa = df['heavy_chain_aa'].values
        blk = df['block_id'].values
        hfam = np.array([g.split('-')[0] for g in hv])
        flv = lv.reshape(-1)
        fla = aas.reshape(-1)
        flc = lc.reshape(-1)
        rep = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], 8, 1)
        F = []

        for wi in range(len(SHM_WINDOWS)):
            wk = SHM_WINDOWS[wi]
            hrate, hcnt, hcov = self.hcons[wk].mut(hv, haa, hc3)
            lrate, lcnt, lcov = self.lcons[wk].mut(flv, fla, flc)
            lrate = lrate.reshape(n, 8)
            lcnt = lcnt.reshape(n, 8)
            lcov = lcov.reshape(n, 8)
            for h, l in ((hrate, lrate), (hcnt, lcnt)):
                hrk = _rank(h, blk)
                lrk = _crank(l)
                hzz = _z(h, blk)
                lzz = _cz(l)
                F += [-np.abs(hrk[:, None] - lrk), -np.abs(hzz[:, None] - lzz), hzz[:, None] * lzz,
                      -np.abs(rep(h) - l), rep(h) - l, l, lzz, rep(h), rep(hzz)]
            if wi == 0:
                F.append(lcov / 100.0)
                hp = self._pct_ref(self.href, self.hall, hv, hrate)
                lp = self._pct_ref(self.lref, self.lall, flv, lrate.reshape(-1)).reshape(n, 8)
                hprk = _rank(hp, blk)
                lprk = _crank(lp)
                hpz = _z(hp, blk)
                lpz = _cz(lp)
                F += [-np.abs(hprk[:, None] - lprk), -np.abs(hpz[:, None] - lpz),
                      hpz[:, None] * lpz, -np.abs(rep(hp) - lp), rep(hp) - lp, lp, lpz,
                      rep(hp), rep(hpz)]

        lfam = np.array([[g.split('-')[0] for g in row] for row in lv], dtype=object)
        loc = np.array([[g[:3] for g in row] for row in lv], dtype=object)
        hb = np.clip(pd.Series(hc3).str.len().values // 3, 0, 9)
        Bm = dict(vv=lv, vfam=lfam, famv=lv, famfam=lfam, vloc=loc, famloc=loc, jj=lj, jv=lv,
                  vj=lj, jloc=loc, dv=lfam, lenv=lv, lenloc=loc)
        Am = dict(vv=hv, vfam=hv, famv=hfam, famfam=hfam, vloc=hv, famloc=hfam, jj=hj, jv=hj,
                  vj=hv, jloc=hj, dv=hd, lenv=hb, lenloc=hb)
        yy = D['y'] if loo else None
        for key, _ in LIFT_SPECS:
            t = self.T[key]
            Ak = Am[key]
            Bk = Bm[key]
            if loo:
                F.append(np.array([[t.loo(Ak[i], Bk[i][yy[i]], Bk[i][j]) for j in range(8)]
                                   for i in range(n)], np.float32))
            else:
                F.append(np.array([[t(Ak[i], Bk[i][j]) for j in range(8)]
                                   for i in range(n)], np.float32))

        hp_ = np.array([seq_props(s) for s in hc3])
        lp_ = np.array([seq_props(s) for s in flc]).reshape(n, 8, -1)
        for k in range(hp_.shape[1]):
            F += [lp_[:, :, k], -np.abs(rep(hp_[:, k]) - lp_[:, :, k]),
                  rep(hp_[:, k]) * lp_[:, :, k]]
        F += self._clonal(blk, hc3, hv, lc, lv)
        X = np.stack(F, -1).astype(np.float32)

        V = self.vocab
        cat_h = np.stack([np.array([V['hv'].get(x, 0) for x in hv]),
                          np.array([V['hj'].get(x, 0) for x in hj]),
                          np.array([V['hd'].get(x, 0) for x in hd]),
                          np.array([V['sp'].get(x, 0) for x in df['species'].values])], 1)
        cat_l = np.stack([np.array([[V['lv'].get(x, 0) for x in row] for row in lv]),
                          np.array([[V['lj'].get(x, 0) for x in row] for row in lj])], -1)
        return dict(X=X, cat_h=cat_h.astype(np.int64), cat_l=cat_l.astype(np.int64), block=blk)


# ------------------------------------------------------------------ pretrained LM tower
@torch.no_grad()
def lm_embed(repo, tok_cls, seqs, spans):
    """Two pooled views per sequence: the whole chain, and the CDR3 span alone.

    Mean-pooling 120 residues is dominated by framework, which is germline-determined and
    shared by every candidate with the same V gene. Adding a CDR3-only pool is worth +0.008
    OOF adjusted (0.5657 -> 0.5739) on the same folds. Residue r sits at token r+1 (one [CLS]).
    """
    tok = tok_cls.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(DEVICE).eval()
    hdim = model.config.hidden_size
    out = np.zeros((len(seqs), hdim * 2), np.float32)
    ar = torch.arange(LM_MAXLEN, device=DEVICE)[None, :]
    for i in range(0, len(seqs), LM_BATCH):
        chunk = [' '.join(list(s)) for s in seqs[i:i + LM_BATCH]]
        enc = tok(chunk, return_tensors='pt', padding='max_length', truncation=True,
                  max_length=LM_MAXLEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(**enc).last_hidden_state.float()
        m = enc['attention_mask'].unsqueeze(-1).float()
        sp = torch.tensor(spans[i:i + len(chunk)], device=DEVICE)
        cm = ((ar >= sp[:, 0:1] + 1) & (ar < sp[:, 1:2] + 1)).unsqueeze(-1).float()
        pa = (h * m).sum(1) / m.sum(1)
        pc = (h * cm).sum(1) / cm.sum(1).clamp(min=1)
        out[i:i + len(chunk)] = torch.cat([pa, pc], 1).cpu().numpy()
        if i % (LM_BATCH * 100) == 0:
            log(f'    {repo} {i}/{len(seqs)}')
    del model
    torch.cuda.empty_cache()
    return out


def standardise(E, fit_rows):
    """Per-dimension standardisation fitted on the TRAINING sequences only. Using the pooled
    train+test statistics would make inference transductive for no measurable gain."""
    sub = E[fit_rows]
    return (E - sub.mean(0, keepdims=True)) / (sub.std(0, keepdims=True) + 1e-5)


# ---------------------------------------------------------------------------- model
def log_sinkhorn(logits, n_iter, tau):
    z = logits / tau
    for _ in range(n_iter):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return z


class BlockMatcher(nn.Module):
    def __init__(self, n_feat, vocab, EH, EL):
        super().__init__()
        d = DGENE
        edim = EH.shape[1]
        # Held as indexed buffers, not materialised per pair: a block's 8 candidates repeat
        # across its 8 rows, so a (N,8,edim) light tensor would be ~3 GB for nothing.
        self.register_buffer('EH', torch.tensor(EH, dtype=torch.float32))
        self.register_buffer('EL', torch.tensor(EL, dtype=torch.float32))
        self.e_hv = nn.Embedding(vocab['hv'] + 1, d)
        self.e_hj = nn.Embedding(vocab['hj'] + 1, 12)
        self.e_hd = nn.Embedding(vocab['hd'] + 1, 12)
        self.e_sp = nn.Embedding(vocab['sp'] + 1, 6)
        self.e_lv = nn.Embedding(vocab['lv'] + 1, d)
        self.e_lj = nn.Embedding(vocab['lj'] + 1, 12)
        for e in (self.e_hv, self.e_hj, self.e_hd, self.e_sp, self.e_lv, self.e_lj):
            nn.init.normal_(e.weight, 0, 0.05)
        self.ehn = nn.LayerNorm(edim)
        self.eln = nn.LayerNorm(edim)
        self.edrop = nn.Dropout(EDROP)
        self.ph = nn.Linear(edim, DEMB)
        self.pl = nn.Linear(edim, DEMB)
        self.bil_e = nn.Parameter(torch.zeros(DEMB, DEMB))
        self.bn = nn.BatchNorm1d(n_feat)
        din = n_feat + d * 3 + 12 * 3 + 6 + DEMB * 3
        self.mlp = nn.Sequential(
            nn.Linear(din, HID), nn.GELU(), nn.Dropout(PDROP),
            nn.Linear(HID, HID // 2), nn.GELU(), nn.Dropout(PDROP),
            nn.Linear(HID // 2, 1))
        self.bil = nn.Parameter(torch.zeros(d, d))
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, X, cat_h, cat_l, ih, il, sinkhorn_iter):
        eh_raw = self.EH[ih]
        el_raw = self.EL[il]
        N = X.shape[0]
        xf = self.bn(X.reshape(N * 8, -1)).reshape(N, 8, -1)
        hv = self.e_hv(cat_h[:, 0])
        hj = self.e_hj(cat_h[:, 1])
        hd = self.e_hd(cat_h[:, 2])
        sp = self.e_sp(cat_h[:, 3])
        lv = self.e_lv(cat_l[..., 0])
        lj = self.e_lj(cat_l[..., 1])
        hvb = hv.unsqueeze(1).expand(-1, 8, -1)
        eh = self.ph(self.edrop(self.ehn(eh_raw)))
        el = self.pl(self.edrop(self.eln(el_raw)))
        ehb = eh.unsqueeze(1).expand(-1, 8, -1)
        z = torch.cat([xf, hvb, lv, hvb * lv,
                       hj.unsqueeze(1).expand(-1, 8, -1),
                       hd.unsqueeze(1).expand(-1, 8, -1),
                       lj, sp.unsqueeze(1).expand(-1, 8, -1),
                       ehb, el, ehb * el], dim=-1)
        s = self.mlp(z).squeeze(-1)
        s = s + torch.einsum('nd,de,nke->nk', hv, self.bil, lv)
        s = s + torch.einsum('nd,de,nke->nk', eh, self.bil_e, el)
        B = s.shape[0] // 8
        M = s.reshape(B, 8, 8)
        return M, log_sinkhorn(M, sinkhorn_iter, 1.0 + Fn.softplus(self.tau))


# ------------------------------------------------------------- exact permutation marginals
def _subsets(n):
    idx = np.arange(1, 1 << n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    return bits, (-1.0) ** (n - bits.sum(1))


def perm_batch(A):
    B, n, _ = A.shape
    bits, signs = _subsets(n)
    rowsum = np.einsum('bij,sj->bsi', A, bits)
    return np.prod(rowsum, axis=2) @ signs


def marginals(M, temp):
    """M (B,8,8) log-potentials -> exact P(pi(i)=j) under the permutation model."""
    B, n, _ = M.shape
    Z = M / temp
    Z = Z - Z.max(axis=(1, 2), keepdims=True)
    W = np.exp(Z).astype(np.float64)
    W = W / np.maximum(W.mean(axis=(1, 2), keepdims=True), 1e-300)
    out = np.zeros((B, n, n), dtype=np.float64)
    ri = np.arange(n)
    for i in range(n):
        Wr = W[:, ri != i, :]
        for j in range(n):
            out[:, i, j] = W[:, i, j] * perm_batch(np.ascontiguousarray(Wr[:, :, ri != j]))
    return out / np.maximum(out.sum(axis=2, keepdims=True), 1e-300)


# ---------------------------------------------------------------------------- driver
def main():
    data_dir = sys.argv[1]
    out_path = sys.argv[2]
    log(f'data_dir={data_dir} out={out_path}')

    TR = load_split(os.path.join(data_dir, 'train.csv'), True)
    TE = load_split(os.path.join(data_dir, 'test.csv'), False)
    log(f'train={TR["df"].shape} test={TE["df"].shape}')

    heavy = sorted(set(TR['df']['heavy_chain_aa'].astype(str)) |
                   set(TE['df']['heavy_chain_aa'].astype(str)))
    light = sorted(set(TR['aas'].reshape(-1).tolist()) | set(TE['aas'].reshape(-1).tolist()))
    log(f'unique heavy={len(heavy)} light={len(light)}')
    hi = {s: i for i, s in enumerate(heavy)}
    li = {s: i for i, s in enumerate(light)}

    def emb_idx(D):
        ih = np.array([hi[s] for s in D['df']['heavy_chain_aa'].astype(str)], np.int64)
        il = np.array([[li[s] for s in row] for row in D['aas']], np.int64)
        return ih, il

    ih_tr, il_tr = emb_idx(TR)
    ih_te, il_te = emb_idx(TE)
    rows_h = np.unique(ih_tr)
    rows_l = np.unique(il_tr.reshape(-1))
    # CDR3 spans in residue coordinates, for the region-pooled view
    hcd = {}
    lcd = {}
    for Dx in (TR, TE):
        hcd.update(dict(zip(Dx['df']['heavy_chain_aa'].astype(str), Dx['df']['heavy_cdr3_aa'])))
        for i in range(len(Dx['df'])):
            for j in range(8):
                lcd[Dx['aas'][i, j]] = Dx['lc'][i, j]
    h_span = np.array([[max(s.rfind(hcd[s]), 0), max(s.rfind(hcd[s]), 0) + len(hcd[s])]
                       for s in heavy])
    l_span = np.array([[max(s.rfind(lcd[s]), 0), max(s.rfind(lcd[s]), 0) + len(lcd[s])]
                       for s in light])

    eh_parts = []
    el_parts = []
    for repo, tok_cls in LM_SPECS:
        log(f'  encoding with {repo}')
        eh_parts.append(standardise(lm_embed(repo, tok_cls, heavy, h_span), rows_h))
        el_parts.append(standardise(lm_embed(repo, tok_cls, light, l_span), rows_l))
    EH_ALL = np.concatenate(eh_parts, axis=1)
    EL_ALL = np.concatenate(el_parts, axis=1)
    del eh_parts, el_parts
    log(f'embeddings {EH_ALL.shape} {EL_ALL.shape}')

    fb = FeatureBuilder().fit(TR)
    Ftr = fb.transform(TR, loo=True)
    Fte = fb.transform(TE, loo=False)
    vocab = {k: len(v) for k, v in fb.vocab.items()}
    log(f'features train={Ftr["X"].shape} test={Fte["X"].shape}')

    otr, Btr = block_order(Ftr['block'])
    ote, Bte = block_order(Fte['block'])

    T = lambda a, o: torch.tensor(a[o], device=DEVICE)
    Xt, Ht, Lt = T(Ftr['X'], otr), T(Ftr['cat_h'], otr), T(Ftr['cat_l'], otr)
    Yt = T(TR['y'], otr)
    IHt, ILt = T(ih_tr, otr), T(il_tr, otr)
    Xv, Hv, Lv = T(Fte['X'], ote), T(Fte['cat_h'], ote), T(Fte['cat_l'], ote)
    IHv, ILv = T(ih_te, ote), T(il_te, ote)

    acc = np.zeros((len(TE['df']), 8), np.float64)
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = BlockMatcher(Xt.shape[-1], vocab, EH_ALL, EL_ALL).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        steps = EPOCHS * ((Btr + BLOCKS_PER_BATCH - 1) // BLOCKS_PER_BATCH)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.25)
        g = torch.Generator().manual_seed(seed)
        for ep in range(EPOCHS):
            m.train()
            perm = torch.randperm(Btr, generator=g)
            for k in range(0, Btr, BLOCKS_PER_BATCH):
                bi = perm[k:k + BLOCKS_PER_BATCH]
                ri = (bi[:, None] * 8 + torch.arange(8)).reshape(-1).to(DEVICE)
                M, _ = m(Xt[ri], Ht[ri], Lt[ri], IHt[ri], ILt[ri], SINK_TRAIN)
                tgt = Yt[ri].reshape(-1, 8)
                loss = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1)) + \
                    COLW * Fn.cross_entropy(M.transpose(1, 2).reshape(-1, 8),
                                            torch.argsort(tgt, 1).reshape(-1))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                sch.step()
        m.eval()
        with torch.no_grad():
            Mr, _ = m(Xv, Hv, Lv, IHv, ILv, 30)
        acc += Mr.reshape(-1, 8).double().cpu().numpy()[np.argsort(ote)]
        log(f'  seed {seed} done')
    raw = acc / N_SEEDS

    p = marginals(raw[ote].reshape(Bte, 8, 8), MARGINAL_T)
    scores = p.reshape(-1, 8)[np.argsort(ote)]
    order = np.argsort(-scores, axis=1, kind='stable') + 1
    rankings = [';'.join(str(int(v)) for v in row) for row in order]
    sub = pd.DataFrame({'id': TE['df']['id'].values, 'ranking': rankings})
    sub.to_csv(out_path, index=False)
    log(f'wrote {out_path} rows={len(sub)}')


main()

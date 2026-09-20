"""Probabilistic, windowed somatic-hypermutation estimation.

Improvement over BackConsensus: instead of hamming distance to the modal residue, score
-log P(residue | gene, position) from the empirical per-(gene,position) residue distribution.
That (a) does not punish allelic polymorphism as if it were mutation, and (b) weights a rare
substitution more than a common one. Counts are reported per *window* of backwards positions so
the downstream model can pick which region tracks maturation best (framework mutations are the
cleaner clock; CDR mutations are selection-driven).
"""
import numpy as np, collections

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}
GAP = 20
NA = 21


def encode_back(seqs, cdr3s, w):
    out = np.full((len(seqs), w), GAP, dtype=np.int8)
    for i, (s, c) in enumerate(zip(seqs, cdr3s)):
        p = s.rfind(c)
        if p < 0:
            continue
        k = min(w, p)
        out[i, :k] = [AA2I.get(s[p - 1 - j], GAP) for j in range(k)]
    return out


class ProbGermline:
    def __init__(self, w=112, min_n=6, alpha=1.0, windows=((0, 16), (16, 32), (32, 48),
                                                           (48, 72), (72, 112), (0, 112))):
        self.w = w; self.min_n = min_n; self.alpha = alpha; self.windows = windows

    def _dist(self, enc):
        """(w,21) probability table with Dirichlet(alpha) smoothing; GAP excluded."""
        t = np.zeros((self.w, NA), dtype=np.float64)
        for p in range(self.w):
            col = enc[:, p]
            col = col[col != GAP]
            if len(col):
                t[p, :] = np.bincount(col, minlength=NA)
        t[:, GAP] = 0.0
        t = t + self.alpha
        return t / t.sum(1, keepdims=True)

    def fit(self, genes, seqs, cdr3s):
        enc = encode_back(seqs, cdr3s, self.w)
        self.glob = self._dist(enc)
        self.tab = {}
        by = collections.defaultdict(list)
        for i, g in enumerate(genes):
            by[g].append(i)
        for g, idx in by.items():
            if len(idx) >= self.min_n:
                # blend toward global so a thin gene is not over-confident
                d = self._dist(enc[idx])
                lam = len(idx) / (len(idx) + 12.0)
                self.tab[g] = lam * d + (1 - lam) * self.glob
        return self

    def score(self, genes, seqs, cdr3s):
        """Return (n, n_windows*3) features: surprisal sum, expected-mismatch sum, coverage."""
        enc = encode_back(seqs, cdr3s, self.w)
        n = len(enc)
        nw = len(self.windows)
        out = np.zeros((n, nw * 3), dtype=np.float32)
        # group rows by gene so the table lookup is vectorised
        by = collections.defaultdict(list)
        for i, g in enumerate(genes):
            by[g].append(i)
        for g in sorted(by.keys()):
            idx = np.array(by[g])
            T = self.tab.get(g, self.glob)                      # (w,21)
            e = enc[idx]                                        # (m,w)
            m = e != GAP
            pr = np.where(m, T[np.arange(self.w)[None, :], np.clip(e, 0, NA - 1)], 1.0)
            surp = np.where(m, -np.log(np.clip(pr, 1e-6, 1.0)), 0.0)
            mism = np.where(m, 1.0 - pr, 0.0)
            for k, (a, b) in enumerate(self.windows):
                cov = m[:, a:b].sum(1).astype(np.float32)
                out[idx, k * 3 + 0] = surp[:, a:b].sum(1)
                out[idx, k * 3 + 1] = mism[:, a:b].sum(1)
                out[idx, k * 3 + 2] = cov
        return out

    def residues(self, seqs, cdr3s, p=24):
        """Residue identity at the p backwards positions nearest CDR3 (int64, 0..20)."""
        return encode_back(seqs, cdr3s, p).astype(np.int64)

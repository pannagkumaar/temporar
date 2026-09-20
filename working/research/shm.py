"""CDR3-anchored positional germline consensus -> somatic-hypermutation proxy.

Sequences are trimmed differently by different studies, so every position is indexed
*backwards from the CDR3 start* (and forwards from the CDR3 end for FR4). That makes the
coordinate system independent of 5' trimming.
"""
import numpy as np, collections

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}
GAP = 20


def encode_back(seq, cdr3, w):
    """Return int8 array of length w: index k = residue at (cdr3_start - 1 - k). GAP if absent."""
    p = seq.rfind(cdr3)
    out = np.full(w, GAP, dtype=np.int8)
    if p < 0:
        return out
    for k in range(min(w, p)):
        out[k] = AA2I.get(seq[p - 1 - k], GAP)
    return out


class BackConsensus:
    """Per-gene consensus in CDR3-anchored backwards coordinates."""

    def __init__(self, w=90, min_n=4):
        self.w = w; self.min_n = min_n
        self.cons = {}; self.glob = None; self.cov = {}

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
        enc = np.stack([encode_back(s, c, self.w) for s, c in zip(seqs, cdr3s)])
        self.glob = self._cons(enc)
        buckets = collections.defaultdict(list)
        for i, g in enumerate(genes):
            buckets[g].append(i)
        for g, idx in buckets.items():
            if len(idx) >= self.min_n:
                self.cons[g] = self._cons(enc[idx])
        return self

    def mut(self, genes, seqs, cdr3s):
        """Return (rate, count, covered) arrays."""
        enc = np.stack([encode_back(s, c, self.w) for s, c in zip(seqs, cdr3s)])
        n = len(enc)
        rate = np.zeros(n); cnt = np.zeros(n); cov = np.zeros(n)
        for i in range(n):
            c = self.cons.get(genes[i])
            if c is None:
                c = self.glob
            m = (enc[i] != GAP) & (c != GAP)
            k = int(m.sum())
            cov[i] = k
            if k > 0:
                d = int((enc[i][m] != c[m]).sum())
                cnt[i] = d
                rate[i] = d / k
        return rate, cnt, cov

"""Two-anchor coordinate system for germline consensus.

Anchoring everything backwards from CDR3 aligns FR3 well, but CDR1/CDR2 length polymorphism
shifts the frame further back, so the distant windows measured weakly (win(72,112) adj=0.080 vs
win(48,72) adj=0.211). A second anchor on the conserved FR2 tryptophan (IMGT 41; heavy
'W[VILMF][RKQ]Q', light 'W[YFH][QLR]Q') re-aligns everything upstream of CDR2.
"""
import re, numpy as np, collections

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}
GAP = 20

H_MOTIF = re.compile(r'W[VILMFA][RKQNG]Q')
L_MOTIF = re.compile(r'W[YFH][QLRK][QKL]')


def find_anchor(seq, cdr3, heavy):
    """Index of the conserved FR2 W, searched only upstream of CDR3."""
    p = seq.rfind(cdr3)
    lim = p if p > 0 else len(seq)
    rx = H_MOTIF if heavy else L_MOTIF
    best = -1
    for m in rx.finditer(seq[:lim]):
        best = m.start()
    if best < 0:
        i = seq.rfind('W', 0, max(lim - 20, 0))
        best = i
    return best


def encode_tracks(seqs, cdr3s, heavy, w_back=48, w_fwd=26, w_up=34):
    """Three aligned tracks per sequence:
       A: w_back residues backwards from CDR3 start   (FR3)
       B: w_fwd residues forwards from the FR2 anchor (FR2 -> CDR2 start)
       C: w_up residues backwards from the FR2 anchor (CDR1 + FR1 tail)
    """
    n = len(seqs)
    A = np.full((n, w_back), GAP, np.int8)
    Bt = np.full((n, w_fwd), GAP, np.int8)
    C = np.full((n, w_up), GAP, np.int8)
    for i, (s, c) in enumerate(zip(seqs, cdr3s)):
        p = s.rfind(c)
        if p > 0:
            for k in range(min(w_back, p)):
                A[i, k] = AA2I.get(s[p - 1 - k], GAP)
        a = find_anchor(s, c, heavy)
        if a >= 0:
            for k in range(min(w_fwd, len(s) - a)):
                Bt[i, k] = AA2I.get(s[a + k], GAP)
            for k in range(min(w_up, a)):
                C[i, k] = AA2I.get(s[a - 1 - k], GAP)
    return A, Bt, C


class TwoAnchor:
    def __init__(self, heavy, min_n=4, w_back=48, w_fwd=26, w_up=34):
        self.heavy = heavy; self.min_n = min_n
        self.dims = (w_back, w_fwd, w_up)

    @staticmethod
    def _cons(arr):
        out = np.full(arr.shape[1], GAP, np.int8)
        for p in range(arr.shape[1]):
            col = arr[:, p]; col = col[col != GAP]
            if len(col) >= 3:
                out[p] = np.bincount(col, minlength=21).argmax()
        return out

    def fit(self, genes, seqs, cdr3s):
        self.tracks = encode_tracks(seqs, cdr3s, self.heavy, *self.dims)
        self.glob = [self._cons(t) for t in self.tracks]
        by = collections.defaultdict(list)
        for i, g in enumerate(genes):
            by[g].append(i)
        self.cons = {}
        for g, idx in by.items():
            if len(idx) >= self.min_n:
                self.cons[g] = [self._cons(t[idx]) for t in self.tracks]
        return self

    def mut(self, genes, seqs, cdr3s):
        """Return (n, 3*2) = per-track (count, coverage), plus a combined count/rate pair."""
        tr = encode_tracks(seqs, cdr3s, self.heavy, *self.dims)
        n = len(seqs)
        out = np.zeros((n, 8), np.float32)
        by = collections.defaultdict(list)
        for i, g in enumerate(genes):
            by[g].append(i)
        for g in sorted(by.keys()):
            idx = np.array(by[g])
            cs = self.cons.get(g, self.glob)
            tot_d = np.zeros(len(idx)); tot_k = np.zeros(len(idx))
            for t in range(3):
                e = tr[t][idx]; c = cs[t]
                m = (e != GAP) & (c != GAP)[None, :]
                d = ((e != c[None, :]) & m).sum(1).astype(np.float64)
                k = m.sum(1).astype(np.float64)
                out[idx, t * 2] = d
                out[idx, t * 2 + 1] = k
                tot_d += d; tot_k += k
            out[idx, 6] = tot_d
            out[idx, 7] = tot_d / np.maximum(tot_k, 1.0)
        return out

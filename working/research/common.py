"""Shared loading / feature / scoring helpers for the light-chain pairing task."""
import numpy as np, pandas as pd, collections

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA2I = {c: i for i, c in enumerate(AA)}


def load(path, with_target=True):
    df = pd.read_csv(path)
    n = len(df)
    codes = df[[f'cand{i}_code' for i in range(1, 9)]].values.astype(object)
    aas = df[[f'cand{i}_aa' for i in range(1, 9)]].values.astype(object)
    lv = np.empty((n, 8), dtype=object); lj = np.empty((n, 8), dtype=object)
    lc = np.empty((n, 8), dtype=object)
    for i in range(n):
        for j in range(8):
            a, b, c = codes[i, j].split('|')
            lv[i, j] = a; lj[i, j] = b; lc[i, j] = c
    df['heavy_d_gene'] = df['heavy_d_gene'].fillna('NA')
    return dict(df=df, codes=codes, aas=aas, lv=lv, lj=lj, lc=lc,
                y=(df['true_candidate_index'].values - 1) if with_target else None)


# ---------- metric ----------
def credit_from_scores(S, y):
    """S (n,8) higher=better. rank of true = 1 + #cands strictly better (ties -> stable by index)."""
    order = np.argsort(-S, axis=1, kind='stable')
    rank = np.empty(len(S), dtype=np.int64)
    pos = np.argsort(order, axis=1, kind='stable')
    rank = pos[np.arange(len(S)), y] + 1
    return (8 - rank) / 7.0


def adjusted(credit):
    return float(np.clip(2 * credit.mean() - 1, 0, 1))


def report(S, y, groups=None, label=''):
    c = credit_from_scores(S, y)
    out = f'{label}: adjusted={adjusted(c):.4f}  mean_credit={c.mean():.4f}  top1={(c==1).mean():.4f}'
    return out, c


# ---------- SHM estimation ----------
def fr3_window(seq, cdr3, w=40):
    """The w residues immediately preceding CDR3 (robust to 5' trimming)."""
    p = seq.rfind(cdr3)
    if p < 0:
        return None
    s = seq[max(0, p - w):p]
    return s if len(s) == w else None


def fr4_window(seq, cdr3, w=10):
    p = seq.rfind(cdr3)
    if p < 0:
        return None
    s = seq[p + len(cdr3): p + len(cdr3) + w]
    return s if len(s) == w else None


class ConsensusSHM:
    """Per-V-gene positional consensus over a CDR3-anchored window; hamming = SHM proxy."""

    def __init__(self, w=40):
        self.w = w
        self.cons = {}
        self.glob = None

    def fit(self, genes, seqs, cdr3s):
        buckets = collections.defaultdict(list)
        allw = []
        for g, s, c in zip(genes, seqs, cdr3s):
            win = fr3_window(s, c, self.w)
            if win is not None:
                buckets[g].append(win)
                allw.append(win)
        self.glob = self._consensus(allw)
        for g, ws in buckets.items():
            if len(ws) >= 5:
                self.cons[g] = self._consensus(ws)
        return self

    def _consensus(self, ws):
        if not ws:
            return None
        arr = np.array([[AA2I.get(ch, 20) for ch in w] for w in ws], dtype=np.int8)
        out = []
        for p in range(arr.shape[1]):
            cnt = np.bincount(arr[:, p], minlength=21)
            out.append(int(cnt.argmax()))
        return np.array(out, dtype=np.int8)

    def score(self, gene, seq, cdr3):
        win = fr3_window(seq, cdr3, self.w)
        if win is None:
            return np.nan
        c = self.cons.get(gene, self.glob)
        if c is None:
            return np.nan
        a = np.array([AA2I.get(ch, 20) for ch in win], dtype=np.int8)
        return float((a != c).mean())

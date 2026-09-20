"""Sharpen the maturation clock. Three candidate fixes over plain hamming-to-modal-consensus:
 (b) two-pass consensus  - rebuild the germline from the least-mutated half of each gene's
     sequences, so a mutation hotspot common in a hypermutated repertoire is not adopted as germline;
 (c) per-gene percentile - different V genes have different allelic diversity, so a raw count of
     3 means different things for IGHV3-23 and IGHV1-69. Rank each chain within its own gene;
 (d) both.
"""
import sys, numpy as np, pandas as pd, collections
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from shm import BackConsensus, encode_back, GAP

D = load('train.csv'); df = D['df']; lv = D['lv']; lc = D['lc']; aas = D['aas']; y = D['y']
n = len(df); r = np.arange(n)
tv = lv[r, y]; tc = lc[r, y]; ta = aas[r, y]
blk = df['block_id'].values
hv = df['heavy_v_gene'].values; haa = df['heavy_chain_aa'].values; hc3 = df['heavy_cdr3_aa'].values
flv = lv.reshape(-1); fla = aas.reshape(-1); flc = lc.reshape(-1)
W = 110


def rankmatch(h, l):
    hr = pd.Series(h).groupby(pd.Series(blk)).rank(method='average').values - 1.0
    lr = np.argsort(np.argsort(l, 1, kind='stable'), 1).astype(float)
    return -np.abs(hr[:, None] - lr)


class TwoPass(BackConsensus):
    def fit2(self, genes, seqs, cdr3s, keep=0.5):
        self.fit(genes, seqs, cdr3s)
        rate, _, _ = self.mut(genes, seqs, cdr3s)
        enc = np.stack([encode_back(s, c, self.w) for s, c in zip(seqs, cdr3s)])
        by = collections.defaultdict(list)
        for i, g in enumerate(genes):
            by[g].append(i)
        for g, idx in by.items():
            idx = np.array(idx)
            if len(idx) >= self.min_n:
                k = max(3, int(len(idx) * keep))
                sel = idx[np.argsort(rate[idx], kind='stable')[:k]]
                self.cons[g] = self._cons(enc[sel])
        return self


def pct_within(gene, val):
    s = pd.Series(val)
    return s.groupby(pd.Series(gene)).rank(pct=True).values


# (a) baseline
hb = BackConsensus(W).fit(hv, haa, hc3); lb = BackConsensus(W).fit(tv, ta, tc)
h_a, hcnt_a, _ = hb.mut(hv, haa, hc3)
l_a, lcnt_a, _ = lb.mut(flv, fla, flc); l_a = l_a.reshape(n, 8); lcnt_a = lcnt_a.reshape(n, 8)
print('(a) plain hamming          adj=%.4f' % adjusted(credit_from_scores(rankmatch(h_a, l_a), y)))

# (b) two-pass
hb2 = TwoPass(W).fit2(hv, haa, hc3); lb2 = TwoPass(W).fit2(tv, ta, tc)
h_b, _, _ = hb2.mut(hv, haa, hc3)
l_b, _, _ = hb2.mut(flv, fla, flc)  # placeholder, replaced below
l_b, _, _ = lb2.mut(flv, fla, flc); l_b = l_b.reshape(n, 8)
print('(b) two-pass consensus     adj=%.4f' % adjusted(credit_from_scores(rankmatch(h_b, l_b), y)))

# (c) per-gene percentile on top of (a)
h_c = pct_within(hv, h_a)
l_c = pct_within(flv, l_a.reshape(-1)).reshape(n, 8)
print('(c) per-gene percentile    adj=%.4f' % adjusted(credit_from_scores(rankmatch(h_c, l_c), y)))

# (d) both
h_d = pct_within(hv, h_b)
l_d = pct_within(flv, l_b.reshape(-1)).reshape(n, 8)
print('(d) two-pass + percentile  adj=%.4f' % adjusted(credit_from_scores(rankmatch(h_d, l_d), y)))

# (e) sum of (a) and (c) rank matrices
print('(e) a+c                    adj=%.4f' % adjusted(credit_from_scores(
    rankmatch(h_a, l_a) + rankmatch(h_c, l_c), y)))
# (f) keep fractions for two-pass
for kf in (0.3, 0.7):
    hbx = TwoPass(W).fit2(hv, haa, hc3, kf); lbx = TwoPass(W).fit2(tv, ta, tc, kf)
    hx, _, _ = hbx.mut(hv, haa, hc3); lx, _, _ = lbx.mut(flv, fla, flc); lx = lx.reshape(n, 8)
    print('(f) two-pass keep=%.1f      adj=%.4f' % (kf, adjusted(credit_from_scores(rankmatch(hx, lx), y))))

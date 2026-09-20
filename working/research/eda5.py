"""Is surprisal actually a better maturation clock than hamming-to-consensus?
Single-feature, in-sample, same split-free protocol as eda4 so the numbers are comparable."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted, report
from shm import BackConsensus
from shm2 import ProbGermline

D = load('train.csv'); df = D['df']; lv = D['lv']; lc = D['lc']; aas = D['aas']; y = D['y']
n = len(df); r = np.arange(n)
tv = lv[r, y]; tc = lc[r, y]; ta = aas[r, y]
blk = df['block_id'].values
flv = lv.reshape(-1); fla = aas.reshape(-1); flc = lc.reshape(-1)


def rankmatch(h, l):
    hr = pd.Series(h).groupby(pd.Series(blk)).rank(method='average').values - 1.0
    lr = np.argsort(np.argsort(l, 1, kind='stable'), 1).astype(float)
    return -np.abs(hr[:, None] - lr)


WIN = ((0, 16), (16, 32), (32, 48), (48, 72), (72, 112), (0, 112))
gh = ProbGermline(112, windows=WIN).fit(df['heavy_v_gene'].values, df['heavy_chain_aa'].values,
                                        df['heavy_cdr3_aa'].values)
gl = ProbGermline(112, windows=WIN).fit(tv, ta, tc)
H = gh.score(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
L = gl.score(flv, fla, flc).reshape(n, 8, -1)

print('--- probabilistic, per window ---')
for k, w in enumerate(WIN):
    for off, nm in ((0, 'surprisal'), (1, 'exp-mismatch')):
        h = H[:, k * 3 + off] / np.maximum(H[:, k * 3 + 2], 1.0)
        l = L[:, :, k * 3 + off] / np.maximum(L[:, :, k * 3 + 2], 1.0)
        a = adjusted(credit_from_scores(rankmatch(h, l), y))
        print(f'  win{w} {nm:13s} rank-match adj={a:.4f}')

print('--- hamming baseline ---')
for w in (90, 110, 130):
    hb = BackConsensus(w).fit(df['heavy_v_gene'].values, df['heavy_chain_aa'].values,
                              df['heavy_cdr3_aa'].values)
    lb = BackConsensus(w).fit(tv, ta, tc)
    hr, hc, _ = hb.mut(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
    lr_, lcn, _ = lb.mut(flv, fla, flc)
    lr_ = lr_.reshape(n, 8); lcn = lcn.reshape(n, 8)
    print(f'  w={w} rate rank-match adj={adjusted(credit_from_scores(rankmatch(hr, lr_), y)):.4f}'
          f'   count rank-match adj={adjusted(credit_from_scores(rankmatch(hc, lcn), y)):.4f}')

# best-of-both: average the two rank matrices
hb = BackConsensus(110).fit(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
lb = BackConsensus(110).fit(tv, ta, tc)
hr, hc, _ = hb.mut(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
lrr, lcn, _ = lb.mut(flv, fla, flc); lrr = lrr.reshape(n, 8)
hs = H[:, 5 * 3 + 0] / np.maximum(H[:, 5 * 3 + 2], 1.0)
ls = L[:, :, 5 * 3 + 0] / np.maximum(L[:, :, 5 * 3 + 2], 1.0)
print('combined rank-match adj=%.4f' % adjusted(credit_from_scores(
    rankmatch(hr, lrr) + rankmatch(hs, ls), y)))

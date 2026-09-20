import sys, numpy as np, pandas as pd
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from shm import BackConsensus
from anchor import TwoAnchor, find_anchor

D = load('train.csv'); df = D['df']; lv = D['lv']; lc = D['lc']; aas = D['aas']; y = D['y']
n = len(df); r = np.arange(n)
tv = lv[r, y]; tc = lc[r, y]; ta = aas[r, y]
blk = df['block_id'].values
hv = df['heavy_v_gene'].values; haa = df['heavy_chain_aa'].values; hc3 = df['heavy_cdr3_aa'].values
flv = lv.reshape(-1); fla = aas.reshape(-1); flc = lc.reshape(-1)


def rm(h, l):
    hr = pd.Series(h).groupby(pd.Series(blk)).rank(method='average').values - 1.0
    lr = np.argsort(np.argsort(l, 1, kind='stable'), 1).astype(float)
    return -np.abs(hr[:, None] - lr)


# anchor hit rate
ah = np.array([find_anchor(s, c, True) for s, c in zip(haa[:4000], hc3[:4000])])
al = np.array([find_anchor(s, c, False) for s, c in zip(ta[:4000], tc[:4000])])
print('heavy anchor found %.3f  median pos %.0f' % ((ah >= 0).mean(), np.median(ah[ah >= 0])))
print('light anchor found %.3f  median pos %.0f' % ((al >= 0).mean(), np.median(al[al >= 0])))

# baseline
hb = BackConsensus(110).fit(hv, haa, hc3); lb = BackConsensus(110).fit(tv, ta, tc)
h0, _, _ = hb.mut(hv, haa, hc3); l0, _, _ = lb.mut(flv, fla, flc); l0 = l0.reshape(n, 8)
print('baseline single-anchor      adj=%.4f' % adjusted(credit_from_scores(rm(h0, l0), y)))

ta_h = TwoAnchor(True).fit(hv, haa, hc3)
ta_l = TwoAnchor(False).fit(tv, ta, tc)
H = ta_h.mut(hv, haa, hc3); L = ta_l.mut(flv, fla, flc).reshape(n, 8, 8)
names = ['trackA_FR3', 'trackB_FR2', 'trackC_CDR1']
for t in range(3):
    h = H[:, t * 2] / np.maximum(H[:, t * 2 + 1], 1.0)
    l = L[:, :, t * 2] / np.maximum(L[:, :, t * 2 + 1], 1.0)
    print('  %-12s cov_h=%.1f cov_l=%.1f adj=%.4f' % (
        names[t], H[:, t * 2 + 1].mean(), L[:, :, t * 2 + 1].mean(),
        adjusted(credit_from_scores(rm(h, l), y))))
print('two-anchor combined rate    adj=%.4f' % adjusted(credit_from_scores(rm(H[:, 7], L[:, :, 7]), y)))
print('two-anchor combined count   adj=%.4f' % adjusted(credit_from_scores(rm(H[:, 6], L[:, :, 6]), y)))
print('single + two-anchor sum     adj=%.4f' % adjusted(credit_from_scores(
    rm(h0, l0) + rm(H[:, 7], L[:, :, 7]), y)))
# within-block correlation of the combined clock
t2 = pd.DataFrame({'b': blk, 'h': H[:, 7], 'l': L[r, y, 7]})
print('within-block corr two-anchor %.4f' % np.corrcoef(
    t2.groupby('b')['h'].transform(lambda v: v - v.mean()),
    t2.groupby('b')['l'].transform(lambda v: v - v.mean()))[0, 1])

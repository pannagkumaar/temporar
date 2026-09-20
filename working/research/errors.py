"""Decoded-error analysis of the incumbent. Looking for a structural failure the next
experiment should target, not another hyperparameter."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from model import block_order
from permmarg import marginals
from shm import BackConsensus

D = load('train.csv'); df = D['df'].copy(); y = D['y']; lv = D['lv']; lc = D['lc']; aas = D['aas']
raw = np.load('.mlctl/artifacts/abpair-pool1/rawoof_pool_mean+cdr3.npy')
o, B = block_order(df['block_id'].values); inv = np.argsort(o)
P = marginals(raw[o].reshape(B, 8, 8), 1.25)
sc = np.log(np.maximum(P, 1e-300)).reshape(-1, 8)[inv]
prob = P.reshape(-1, 8)[inv]
n = len(df); r = np.arange(n)
order = np.argsort(-sc, 1, kind='stable')
rank = np.argsort(order, 1)[r, y] + 1
cred = (8 - rank) / 7.0
hum = df['species'].values == 'human'
print('human adjusted %.4f  top1 %.4f  top2 %.4f' %
      (adjusted(cred[hum]), (rank[hum] == 1).mean(), (rank[hum] <= 2).mean()))
print('rank histogram (human):', np.bincount(rank[hum], minlength=9)[1:] / hum.sum())

# 1. Is the model calibrated? Does its own confidence track correctness?
pt = prob[r, order[:, 0]]
q = pd.qcut(pt[hum], 10, labels=False, duplicates='drop')
t = pd.DataFrame({'q': q, 'conf': pt[hum], 'hit': (rank[hum] == 1).astype(float)})
print('\ncalibration of top-1 marginal (human):')
print(t.groupby('q').agg(conf=('conf', 'mean'), acc=('hit', 'mean'), n=('hit', 'size')).round(3).to_string())

# 2. Where does it fail? condition on properties of the TRUE pair
tv = lv[r, y]
df['rank'] = rank; df['cred'] = cred
df['locus'] = [g[:3] for g in tv]
df['n_same_locus'] = [(np.array([x[:3] for x in lv[i]]) == tv[i][:3]).sum() for i in range(n)]
hb = BackConsensus(110).fit(df['heavy_v_gene'].values, df['heavy_chain_aa'].values,
                            df['heavy_cdr3_aa'].values)
hr, _, _ = hb.mut(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
df['hshm'] = hr
h = df[hum]
print('\nby true-partner locus:')
print(h.groupby('locus')['cred'].agg(['size', 'mean']).assign(
    adj=lambda d: (2 * d['mean'] - 1).clip(0, 1)).round(4).to_string())
print('\nby number of block candidates sharing the true locus (harder = more):')
print(h.groupby('n_same_locus')['cred'].agg(['size', 'mean']).assign(
    adj=lambda d: (2 * d['mean'] - 1).clip(0, 1)).round(4).to_string())
print('\nby heavy SHM decile (0 = least mutated / naive):')
h2 = h.assign(d=pd.qcut(h['hshm'], 10, labels=False, duplicates='drop'))
print(h2.groupby('d')['cred'].agg(['size', 'mean']).assign(
    adj=lambda d: (2 * d['mean'] - 1).clip(0, 1)).round(4).to_string())

# 3. Does the block's SHM spread predict difficulty? (the clock needs dispersion to work)
bl = h.groupby('block_id').agg(spread=('hshm', 'std'), cred=('cred', 'mean'))
bl['q'] = pd.qcut(bl['spread'], 5, labels=False, duplicates='drop')
print('\nby within-block heavy-SHM dispersion:')
print(bl.groupby('q').agg(spread=('spread', 'mean'), adj=('cred', lambda v: np.clip(2 * v.mean() - 1, 0, 1)),
                          n=('cred', 'size')).round(4).to_string())

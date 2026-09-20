"""Two free experiments on the delivered model's saved OOF logits.

1. Block-adaptive marginal temperature. The delivered decoder uses one global T=1.25. If the
   model's confidence is better calibrated on some blocks than others, a T that depends on an
   observable block property should beat a constant one. The property must be computable at
   test time from the block alone.

2. What makes a donor hard? The score weights min(family) at 0.25, and per-fold scores span
   0.550-0.622, so whatever separates an easy donor from a hard one is worth 4x its share.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from model import block_order
from permmarg import marginals
from shm import BackConsensus

ART = '.mlctl/artifacts/abpair-conf1'
D = load('train.csv'); df = D['df'].copy(); y = D['y']
raw = np.load(f'{ART}/rawoof_s16_edrop45.npy')
o, B = block_order(df['block_id'].values); inv = np.argsort(o)
M = raw[o].reshape(B, 8, 8)
hum = df['species'].values == 'human'


def sc_from(p):
    return np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]


print('=== 1. marginal temperature ===')
best = (0, None)
for T in (0.9, 1.0, 1.1, 1.25, 1.4, 1.6):
    s = sc_from(marginals(M, T))
    c = credit_from_scores(s, y)
    a, h = adjusted(c), adjusted(c[hum])
    print(f'  global T={T:<5} OOF={a:.4f} human={h:.4f}')
    if h > best[0]:
        best = (h, T)
print(f'  best global T={best[1]} human={best[0]:.4f}')

# block-adaptive: T as a function of the block's own logit spread (computable at test time)
spread = M.std(axis=(1, 2))
q = pd.qcut(pd.Series(spread), 5, labels=False, duplicates='drop').values
print('\n  per-quintile of block logit spread, best T each:')
Pbest = np.zeros((B, 8, 8))
chosen = []
for k in sorted(set(q)):
    sel = q == k
    bb = (0, None)
    for T in (0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.9):
        p = marginals(M[sel], T)
        # score only the rows of these blocks
        rows = np.zeros(len(df), bool)
        idx = o.reshape(B, 8)[sel].reshape(-1)
        rows[idx] = True
        sfull = np.zeros((len(df), 8))
        sfull[idx] = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)
        a = adjusted(credit_from_scores(sfull[rows], y[rows]))
        if a > bb[0]:
            bb = (a, T)
    chosen.append(bb[1])
    Pbest[sel] = marginals(M[sel], bb[1])
    print(f'    q{k}: spread={spread[sel].mean():.2f} bestT={bb[1]} adj={bb[0]:.4f}')
c = credit_from_scores(sc_from(Pbest), y)
print(f'  adaptive-T  OOF={adjusted(c):.4f} human={adjusted(c[hum]):.4f}   '
      f'(global best human={best[0]:.4f})  chosen T per quintile: {chosen}')

print('\n=== 2. what makes a donor hard ===')
s = sc_from(marginals(M, 1.25))
df['cred'] = credit_from_scores(s, y)
hb = BackConsensus(110).fit(df['heavy_v_gene'].values, df['heavy_chain_aa'].values,
                            df['heavy_cdr3_aa'].values)
hr, _, _ = hb.mut(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
df['hshm'] = hr
h = df[hum]
per = h.groupby('sample_id').agg(
    n=('cred', 'size'), cred=('cred', 'mean'),
    shm_mean=('hshm', 'mean'), shm_sd=('hshm', 'std'),
    c3=('heavy_cdr3_aa', lambda v: v.str.len().mean()),
    nvgene=('heavy_v_gene', 'nunique'))
per = per[per['n'] >= 100]
per['adj'] = (2 * per['cred'] - 1).clip(0, 1)
# within-block SHM dispersion averaged over the donor's blocks
bl = h.groupby(['sample_id', 'block_id'])['hshm'].std().groupby('sample_id').mean()
per['blk_shm_sd'] = bl
print(f'  {len(per)} human samples, adjusted min {per["adj"].min():.3f} '
      f'med {per["adj"].median():.3f} max {per["adj"].max():.3f}')
print('  correlation of donor adjusted score with:')
for col in ('shm_mean', 'shm_sd', 'blk_shm_sd', 'c3', 'nvgene', 'n'):
    print(f'    {col:12s} r={np.corrcoef(per["adj"], per[col].fillna(0))[0,1]:+.3f}')
print('\n  hardest 5 donors:'); print(per.nsmallest(5, 'adj')[
    ['n', 'adj', 'shm_mean', 'blk_shm_sd', 'c3']].round(3).to_string())
print('\n  easiest 5 donors:'); print(per.nlargest(5, 'adj')[
    ['n', 'adj', 'shm_mean', 'blk_shm_sd', 'c3']].round(3).to_string())

"""How much can the 0.25*min(family) term hurt?

The two hidden test families are coherent groups (one unseen donor from a seen study; one
wholly unseen study), not random samples. A random 5-fold min is therefore an optimistic
proxy. Here the same OOF predictions are re-aggregated over *coherent* sample clusters built
from technical/repertoire fingerprints, which is the closest thing to a study label the train
data offers, and the worst cluster is reported.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from model import block_order
from permmarg import marginals

ART = '.mlctl/artifacts/abpair-ft1'
D = load('train.csv'); df = D['df']; y = D['y']
raw = np.load(f'{ART}/rawoof_antiberta2_igbert_esm2.npy')
o, B = block_order(df['block_id'].values); inv = np.argsort(o)
p = marginals(raw[o].reshape(B, 8, 8), 1.25)
sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
cred = credit_from_scores(sc, y)
print('overall adjusted %.4f' % adjusted(cred))

# fingerprint each sample
df2 = df.copy()
df2['cred'] = cred
df2['hlen'] = df2['heavy_chain_aa'].str.len()
df2['c3len'] = df2['heavy_cdr3_aa'].str.len()
df2['dmiss'] = df2['heavy_d_gene'].eq('NA') | df2['heavy_d_gene'].isna()
g = df2.groupby('sample_id')
fp = pd.DataFrame({
    'n': g.size(), 'hlen_m': g['hlen'].mean(), 'hlen_s': g['hlen'].std(),
    'c3_m': g['c3len'].mean(), 'c3_s': g['c3len'].std(), 'dmiss': g['dmiss'].mean(),
    'cred': g['cred'].mean(), 'species': g['species'].first()})
# V-gene usage profile
vg = pd.crosstab(df2['sample_id'], df2['heavy_v_gene'], normalize='index')
top = vg.sum(0).sort_values(ascending=False).index[:25]
X = np.hstack([
    ((fp[['hlen_m', 'hlen_s', 'c3_m', 'c3_s', 'dmiss']] -
      fp[['hlen_m', 'hlen_s', 'c3_m', 'c3_s', 'dmiss']].mean()) /
     (fp[['hlen_m', 'hlen_s', 'c3_m', 'c3_s', 'dmiss']].std() + 1e-9)).fillna(0).values,
    vg[top].values * 3.0])
big = fp['n'] >= 100
print('samples used', int(big.sum()))

from scipy.cluster.hierarchy import linkage, fcluster
Z = linkage(X[big.values], 'ward')
for k in (4, 6, 8, 12):
    lab = fcluster(Z, k, 'maxclust')
    cl = pd.Series(lab, index=fp.index[big])
    df2['cl'] = df2['sample_id'].map(cl)
    sub = df2.dropna(subset=['cl'])
    per = sub.groupby('cl').apply(lambda d: pd.Series(
        {'n': len(d), 'adj': np.clip(2 * d['cred'].mean() - 1, 0, 1)}))
    worst = per['adj'].min()
    print(f'k={k:2d} clusters  worst-cluster adjusted={worst:.4f}  '
          f'(sizes {sorted(per["n"].astype(int).tolist())})')
    print('      per-cluster: ' + ' '.join(f'{a:.3f}' for a in sorted(per['adj'])))

# worst single sample and the decile spread
ps = df2.groupby('sample_id')['cred'].agg(['mean', 'size'])
ps['adj'] = (2 * ps['mean'] - 1).clip(0, 1)
b = ps[ps['size'] >= 100]
print('per-sample adjusted: min %.3f q10 %.3f med %.3f' %
      (b['adj'].min(), b['adj'].quantile(.1), b['adj'].median()))
# simulate a 2048-row family drawn as whole samples: worst of many random draws
rng = np.random.default_rng(0)
sids = b.index.values
worst = []
for _ in range(400):
    pick = rng.choice(sids, 7, replace=False)
    m = df2['sample_id'].isin(pick)
    worst.append(np.clip(2 * df2.loc[m, 'cred'].mean() - 1, 0, 1))
worst = np.array(worst)
print('random 7-sample family: mean %.4f  p5 %.4f  min %.4f' %
      (worst.mean(), np.quantile(worst, .05), worst.min()))

import sys, numpy as np, pandas as pd, collections
sys.path.insert(0, 'working/research')
from common import *

D = load('train.csv')
df = D['df']; lv=D['lv']; lj=D['lj']; lc=D['lc']; aas=D['aas']; y=D['y']
n = len(df)
rows = np.arange(n)
tlv = lv[rows, y]; tlj = lj[rows, y]; tlc = lc[rows, y]; taa = aas[rows, y]

# ---- SHM ----
hs = ConsensusSHM(40).fit(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
ls = ConsensusSHM(40).fit(tlv, taa, tlc)
h_shm = np.array([hs.score(g,s,c) for g,s,c in zip(df['heavy_v_gene'], df['heavy_chain_aa'], df['heavy_cdr3_aa'])])
l_shm = np.array([ls.score(g,s,c) for g,s,c in zip(tlv, taa, tlc)])
ok = ~(np.isnan(h_shm)|np.isnan(l_shm))
print('SHM coverage %.3f'%ok.mean())
print('heavy shm mean %.4f  light shm mean %.4f'%(np.nanmean(h_shm), np.nanmean(l_shm)))
print('raw corr %.4f'%np.corrcoef(h_shm[ok], l_shm[ok])[0,1])
s = df['sample_id'].values
tmp = pd.DataFrame({'s':s[ok],'h':h_shm[ok],'l':l_shm[ok]})
tmp['hz']=tmp.groupby('s')['h'].transform(lambda v:v-v.mean())
tmp['lz']=tmp.groupby('s')['l'].transform(lambda v:v-v.mean())
print('within-sample corr %.4f'%np.corrcoef(tmp['hz'],tmp['lz'])[0,1])
# within-BLOCK correlation is what actually matters (decoys come from the block)
tmp2 = pd.DataFrame({'b':df['block_id'].values[ok],'h':h_shm[ok],'l':l_shm[ok]})
tmp2['hz']=tmp2.groupby('b')['h'].transform(lambda v:v-v.mean())
tmp2['lz']=tmp2.groupby('b')['l'].transform(lambda v:v-v.mean())
print('within-block corr %.4f'%np.corrcoef(tmp2['hz'],tmp2['lz'])[0,1])

# ---- single-feature within-block ranking scores (train-fit, in-sample upper bound) ----
sp = df['species'].values; hv = df['heavy_v_gene'].values; hj = df['heavy_j_gene'].values
locus = np.array([[v[:3] for v in row] for row in lv], dtype=object)

def lift_table(keyA, keyB, alpha=20.0):
    """log P(B|A)/P(B) with Dirichlet shrinkage."""
    cab = collections.Counter(zip(keyA, keyB)); ca=collections.Counter(keyA); cb=collections.Counter(keyB)
    N=len(keyA)
    pb = {b: cb[b]/N for b in cb}
    def f(a,b):
        p = pb.get(b, 1e-6)
        num = cab.get((a,b),0) + alpha*p
        den = ca.get(a,0) + alpha
        return np.log((num/den)/p)
    return f

# locus prior
fl = lift_table(hv, np.array([v[:3] for v in tlv]), alpha=20)
S = np.array([[fl(hv[i], locus[i,j]) for j in range(8)] for i in rows])
print(report(S,y,label='locus prior (in-sample)')[0])

# heavy V x light V
fv = lift_table(hv, tlv, alpha=30)
S2 = np.array([[fv(hv[i], lv[i,j]) for j in range(8)] for i in rows])
print(report(S2,y,label='hV x lV lift (in-sample)')[0])
print(report(S+S2,y,label='locus + hVxlV')[0])

# heavy V family x light V
hfam = np.array([g.split('-')[0] for g in hv]); lfam_t=np.array([g.split('-')[0] for g in tlv])
lfam = np.array([[g.split('-')[0] for g in row] for row in lv], dtype=object)
ff = lift_table(hfam, lfam_t, alpha=20)
S3 = np.array([[ff(hfam[i], lfam[i,j]) for j in range(8)] for i in rows])
print(report(S3,y,label='hVfam x lVfam')[0])

# heavy J x light J
fj = lift_table(hj, tlj, alpha=20)
S4 = np.array([[fj(hj[i], lj[i,j]) for j in range(8)] for i in rows])
print(report(S4,y,label='hJ x lJ')[0])

# SHM matching: -(|h_shm - l_shm|)
lsh = np.array([[ls.score(lv[i,j], aas[i,j], lc[i,j]) for j in range(8)] for i in rows])
S5 = -np.abs(h_shm[:,None]-lsh)
S5 = np.nan_to_num(S5, nan=0.0)
print(report(S5,y,label='SHM |diff| (in-sample)')[0])
np.save('working/cache/h_shm.npy', h_shm); np.save('working/cache/l_shm_cands.npy', lsh)

# combine
print(report(S+S2+S4+0.5*S5*10,y,label='combo')[0])

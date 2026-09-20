import pandas as pd, numpy as np, itertools, collections, sys

tr = pd.read_csv('train.csv')

# unpack candidates
codes = tr[[f'cand{i}_code' for i in range(1,9)]].values
aas   = tr[[f'cand{i}_aa'   for i in range(1,9)]].values
ti    = tr['true_candidate_index'].values - 1
n = len(tr)
true_code = np.array([codes[i, ti[i]] for i in range(n)])
true_aa   = np.array([aas[i, ti[i]]   for i in range(n)])
tr['lv'] = [c.split('|')[0] for c in true_code]
tr['lj'] = [c.split('|')[1] for c in true_code]
tr['lcdr3'] = [c.split('|')[2] for c in true_code]
tr['laa'] = true_aa
tr['locus'] = tr['lv'].str[:3]   # IGK / IGL

print('locus dist'); print(tr['locus'].value_counts())
print('n light V genes', tr['lv'].nunique(), 'light J', tr['lj'].nunique())
print('n heavy V genes', tr['heavy_v_gene'].nunique(), 'heavy J', tr['heavy_j_gene'].nunique())

# --- 1. does heavy V predict locus (kappa/lambda)? ---
h = tr[tr.species=='human']
ct = pd.crosstab(h['heavy_v_gene'], h['locus'], normalize='index')
base = (h['locus']=='IGK').mean()
print('\nbase kappa frac %.3f' % base)
sizes = h['heavy_v_gene'].value_counts()
big = sizes[sizes>=200].index
print(ct.loc[[g for g in big if g in ct.index]].sort_values('IGK').head(8))
print(ct.loc[[g for g in big if g in ct.index]].sort_values('IGK').tail(8))

# --- 2. clonal relatives within a block? ---
# heavy cdr3 identical length + high similarity between two rows of the same block
def sim(a,b):
    if len(a)!=len(b): return 0.0
    return sum(x==y for x,y in zip(a,b))/len(a)

cnt_pairs=0; cnt_clonal=0; clonal_light_sim=[]; nonclonal_light_sim=[]
rng = np.random.default_rng(0)
for bid, g in itertools.islice(tr.groupby('block_id'), 1500):
    hs = g['heavy_cdr3_aa'].values; ls = g['lcdr3'].values; lv=g['lv'].values; hv=g['heavy_v_gene'].values
    for i,j in itertools.combinations(range(len(g)),2):
        cnt_pairs+=1
        s = sim(hs[i],hs[j])
        if s>=0.7 and hv[i]==hv[j]:
            cnt_clonal+=1
            clonal_light_sim.append((lv[i]==lv[j], sim(ls[i],ls[j])))
        else:
            nonclonal_light_sim.append((lv[i]==lv[j], sim(ls[i],ls[j])))
print('\npairs',cnt_pairs,'clonal-ish',cnt_clonal, 'rate %.4f'%(cnt_clonal/cnt_pairs))
if clonal_light_sim:
    a=np.array(clonal_light_sim); b=np.array(nonclonal_light_sim)
    print('clonal: same lightV %.3f  mean lcdr3 sim %.3f'%(a[:,0].mean(), a[:,1].mean()))
    print('other : same lightV %.3f  mean lcdr3 sim %.3f'%(b[:,0].mean(), b[:,1].mean()))

# --- 3. heavy V x light V lift (human only) ---
hv = h['heavy_v_gene'].values; lv = h['lv'].values
pair = collections.Counter(zip(hv,lv)); ch=collections.Counter(hv); cl=collections.Counter(lv); N=len(h)
lifts=[]
for (a,b),c in pair.items():
    if c>=25:
        exp = ch[a]*cl[b]/N
        lifts.append((c/exp, c, a, b))
lifts.sort()
print('\nextreme heavy_v x light_v lifts (count>=25):')
for x in lifts[:6]: print('  %.2f  n=%4d  %s / %s'%x)
for x in lifts[-6:]: print('  %.2f  n=%4d  %s / %s'%x)

# --- 4. CDR3 length correlation ---
h2 = h.copy()
h2['hl']=h2['heavy_cdr3_aa'].str.len(); h2['ll']=h2['lcdr3'].str.len()
print('\ncdr3 len corr (raw) %.4f'%np.corrcoef(h2['hl'],h2['ll'])[0,1])
# within-sample residual correlation
h2['hlz']=h2.groupby('sample_id')['hl'].transform(lambda s: s-s.mean())
h2['llz']=h2.groupby('sample_id')['ll'].transform(lambda s: s-s.mean())
print('cdr3 len corr (within sample) %.4f'%np.corrcoef(h2['hlz'],h2['llz'])[0,1])

# charge
def charge(s): return sum(c in 'KR' for c in s)-sum(c in 'DE' for c in s)
h2['hc']=h2['heavy_cdr3_aa'].map(charge); h2['lc']=h2['lcdr3'].map(charge)
h2['hcz']=h2.groupby('sample_id')['hc'].transform(lambda s:s-s.mean())
h2['lcz']=h2.groupby('sample_id')['lc'].transform(lambda s:s-s.mean())
print('cdr3 charge corr (within sample) %.4f'%np.corrcoef(h2['hcz'],h2['lcz'])[0,1])

# --- 5. light chain length (full aa) ---
h2['hlen']=h2['heavy_chain_aa'].str.len(); h2['llen']=h2['laa'].str.len()
print('full len corr within sample %.4f'%np.corrcoef(
    h2.groupby('sample_id')['hlen'].transform(lambda s:s-s.mean()),
    h2.groupby('sample_id')['llen'].transform(lambda s:s-s.mean()))[0,1])

# --- 6. do light chains repeat across blocks within a sample? ---
g = tr.groupby('sample_id')['laa'].agg(['count','nunique'])
print('\nlight aa uniqueness per sample:'); print((g['nunique']/g['count']).describe())

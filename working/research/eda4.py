import sys, numpy as np, pandas as pd, collections
sys.path.insert(0,'working/research')
from common import load, credit_from_scores, adjusted, report
from shm import BackConsensus

D = load('train.csv'); df=D['df']; lv=D['lv']; lc=D['lc']; aas=D['aas']; y=D['y']
n=len(df); rows=np.arange(n)
tlv=lv[rows,y]; tlc=lc[rows,y]; taa=aas[rows,y]

for w in (60, 90, 110):
    hb = BackConsensus(w).fit(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
    lb = BackConsensus(w).fit(tlv, taa, tlc)
    hr,hc,hv_ = hb.mut(df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
    # candidate-side light mutation
    flat_g = lv.reshape(-1); flat_s = aas.reshape(-1); flat_c = lc.reshape(-1)
    lr,lcn,lcv = lb.mut(flat_g, flat_s, flat_c)
    lr=lr.reshape(n,8); lcn=lcn.reshape(n,8); lcv=lcv.reshape(n,8)
    tr_=lr[rows,y]
    b=df['block_id'].values
    t=pd.DataFrame({'b':b,'h':hr,'l':tr_})
    wb=np.corrcoef(t.groupby('b')['h'].transform(lambda v:v-v.mean()), t.groupby('b')['l'].transform(lambda v:v-v.mean()))[0,1]
    t2=pd.DataFrame({'b':b,'h':hc,'l':lcn[rows,y]})
    wb2=np.corrcoef(t2.groupby('b')['h'].transform(lambda v:v-v.mean()), t2.groupby('b')['l'].transform(lambda v:v-v.mean()))[0,1]
    S=-np.abs(hr[:,None]-lr)
    S2=-np.abs(hc[:,None]-lcn)
    print(f'w={w} cov_h={hv_.mean():.1f} cov_l={lcv.mean():.1f} rate_corr={wb:.4f} cnt_corr={wb2:.4f}')
    print('   ', report(S,y,label='  |rate diff|')[0])
    print('   ', report(S2,y,label='  |count diff|')[0])
    # z-scored within block then squared diff
    zh=(hr-hr.mean())/hr.std()
    zl=(lr-lr.mean())/lr.std()
    print('   ', report(-np.abs(zh[:,None]-zl),y,label='  z-rate')[0])
    # rank-based within block
    rl = np.argsort(np.argsort(lr,axis=1),axis=1)   # 0..7 rank of each cand's mutation
    # heavy's rank among the block's heavies
    hh = pd.DataFrame({'b':b,'h':hr}).groupby('b')['h'].rank(method='first').values-1
    print('   ', report(-np.abs(hh[:,None]-rl),y,label='  rank-match')[0])

import pandas as pd, numpy as np, collections, sys

tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv')
for df in (tr,te):
    df['hlen']=df['heavy_chain_aa'].str.len()
    df['hstart']=df['heavy_chain_aa'].str[:4]
    df['hend']=df['heavy_chain_aa'].str[-4:]
    df['dmiss']=df['heavy_d_gene'].isna()

def fp(df, name):
    g = df.groupby('sample_id')
    out = pd.DataFrame({
        'n': g.size(),
        'species': g['species'].first(),
        'hlen_mean': g['hlen'].mean().round(1),
        'hlen_std': g['hlen'].std().round(1),
        'dmiss': g['dmiss'].mean().round(3),
        'top_start': g['hstart'].agg(lambda s: s.value_counts().index[0]),
        'startdiv': g['hstart'].nunique(),
        'top_end': g['hend'].agg(lambda s: s.value_counts().index[0]),
        'enddiv': g['hend'].nunique(),
    })
    print('=== %s fingerprints ==='%name)
    print(out.sort_values(['hlen_mean']).to_string())
    return out

a=fp(tr,'train'); b=fp(te,'test')
print('\ntrain hlen_mean describe'); print(a['hlen_mean'].describe())
print('test hlen_mean describe'); print(b['hlen_mean'].describe())

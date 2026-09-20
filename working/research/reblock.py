"""Within-sample re-blocking augmentation.

The 40 blocks a sample is shipped as are an arbitrary partition of its 320 cells; there are
C(320,8) ~ 2.5e15 equally valid ones. The incumbent therefore trains on ONE frozen arrangement,
seeing 30k positive pairs and 213k negative pairs in fixed company. Resampling the partition
within a sample produces genuinely different blocks with exactly the same structure - 8 real
cells from one donor, a real bijection, same-donor decoys - and no fabricated data.

Candidate slots are rebuilt the way the challenge builds them: sorted alphabetically by
`light_chain_code`, which depends only on the candidate.
"""
import numpy as np


def reblock(D, rows, seed):
    """Return a D-like dict over a re-partitioned copy of `rows` (a training index array)."""
    df = D['df']; lv = D['lv']; lj = D['lj']; lc = D['lc']; aas = D['aas']; y = D['y']
    code = np.array([f'{lv[i, y[i]]}|{lj[i, y[i]]}|{lc[i, y[i]]}' for i in rows], dtype=object)
    aa = np.array([aas[i, y[i]] for i in rows], dtype=object)
    samp = df['sample_id'].values[rows]
    rng = np.random.default_rng(seed)

    order = []
    for s in sorted(set(samp)):
        idx = np.where(samp == s)[0]
        idx = idx[rng.permutation(len(idx))]
        n = (len(idx) // 8) * 8
        order.append(idx[:n])
    if not order:
        return None
    order = np.concatenate(order)
    nb = len(order) // 8
    grp = order.reshape(nb, 8)

    # drop any block that would contain two identical light chains: the target index would be
    # ambiguous and the label noise is not worth the extra block
    keep = np.array([len(set(code[g])) == 8 for g in grp])
    grp = grp[keep]
    nb = len(grp)
    if nb == 0:
        return None
    sel = grp.reshape(-1)                      # positions within `rows`
    abs_rows = rows[sel]

    ncode = np.empty((nb * 8, 8), dtype=object)
    naa = np.empty((nb * 8, 8), dtype=object)
    ny = np.empty(nb * 8, dtype=np.int64)
    for b in range(nb):
        g = grp[b]
        cs = code[g]
        o = np.argsort(cs, kind='stable')      # alphabetical by light_chain_code
        sc = cs[o]; sa = aa[g][o]
        pos = {c: k for k, c in enumerate(sc)}
        for t in range(8):
            r = b * 8 + t
            ncode[r] = sc
            naa[r] = sa
            ny[r] = pos[cs[t]]

    nlv = np.empty_like(ncode); nlj = np.empty_like(ncode); nlc = np.empty_like(ncode)
    for i in range(ncode.shape[0]):
        for j in range(8):
            a, b_, c = ncode[i, j].split('|')
            nlv[i, j] = a; nlj[i, j] = b_; nlc[i, j] = c

    ndf = df.iloc[abs_rows].copy()
    ndf['block_id'] = np.repeat([f'rb{seed}_{b:06d}' for b in range(nb)], 8)
    return dict(df=ndf.reset_index(drop=True), aas=naa, lv=nlv, lj=nlj, lc=nlc, y=ny,
                abs_rows=abs_rows)

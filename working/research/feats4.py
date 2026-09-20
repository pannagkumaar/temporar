"""v4 features = v3 plus block-level SHM dispersion, so the model can gate its own clock.

The gap this closes. The dominant feature family is the within-block rank of somatic mutation
load: heavy rank r vs candidate rank r'. Those ranks are ALWAYS 0..7 whatever the block looks
like. In a block of eight near-germline cells the ranks are pure noise; in a block spanning
0-20% mutation they are highly informative. The incumbent receives the ranks and the z-scores
(which divide the dispersion out) but never the dispersion itself, so it cannot distinguish
the two cases and must apply one fixed amount of trust to both.

Donor-level evidence that this matters: human per-donor adjusted spans 0.425-0.793, and the
hardest donors include two near-naive repertoires (smp_168 mean SHM 0.026, smp_013 0.035)
where the clock has nothing to rank.

Added per block (constant across the row's 8 candidates, which is the point -- it is a gate,
not a discriminator): the spread and level of heavy SHM across the block's 8 heavy chains, the
same for the 8 candidate light chains, and their ratio.
"""
import numpy as np, pandas as pd
from feats3 import FB3, WINDOWS, _rank, _z, _crank, _cz


class FB4(FB3):
    def transform(self, D, idx, loo=False):
        base = super().transform(D, idx, loo=loo)
        df = D['df'].iloc[idx]
        n = len(idx)
        blk = df['block_id'].values
        hv = df['heavy_v_gene'].values
        haa = df['heavy_chain_aa'].values
        hc3 = df['heavy_cdr3_aa'].values
        lv = D['lv'][idx]; lc = D['lc'][idx]; aas = D['aas'][idx]
        flv = lv.reshape(-1); fla = aas.reshape(-1); flc = lc.reshape(-1)

        wk = WINDOWS[0]
        hrate, hcnt, _ = self.hcons[wk].mut(hv, haa, hc3)
        lrate, lcnt, _ = self.lcons[wk].mut(flv, fla, flc)
        lrate = lrate.reshape(n, 8); lcnt = lcnt.reshape(n, 8)

        s = pd.Series(hrate)
        g = s.groupby(pd.Series(blk))
        h_sd = g.transform('std').fillna(0.0).values
        h_mu = g.transform('mean').values
        h_rng = (g.transform('max') - g.transform('min')).values
        l_sd = lrate.std(1)
        l_mu = lrate.mean(1)
        l_rng = lrate.max(1) - lrate.min(1)
        # fraction of the block that is essentially unmutated: the clock is blind there
        h_naive = pd.Series((hrate < 0.02).astype(float)).groupby(pd.Series(blk)).transform('mean').values
        l_naive = (lrate < 0.02).mean(1)

        cols = [h_sd, h_mu, h_rng, l_sd, l_mu, l_rng, h_naive, l_naive,
                np.clip(h_sd / np.maximum(l_sd, 1e-3), 0.0, 10.0), h_sd * l_sd,
                np.minimum(h_sd, l_sd), h_mu * l_mu]
        extra = np.stack([np.repeat(np.asarray(c, np.float32)[:, None], 8, 1) for c in cols], -1)
        base['X'] = np.concatenate([base['X'], extra.astype(np.float32)], axis=-1)
        return base

"""Feature construction: fits everything it needs on a TRAIN fold only, applies to any fold."""
import numpy as np, pandas as pd, collections, sys
from shm import BackConsensus

AA = 'ACDEFGHIKLMNPQRSTVWY'
KD = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
CHG = {c: (1.0 if c in 'KR' else (-1.0 if c in 'DE' else (0.5 if c == 'H' else 0.0))) for c in AA}
AROM = set('FWY')


def seq_props(s):
    n = max(len(s), 1)
    kd = sum(KD.get(c, 0.0) for c in s) / n
    ch = sum(CHG.get(c, 0.0) for c in s)
    ar = sum(c in AROM for c in s) / n
    gly = s.count('G') / n
    ser = (s.count('S') + s.count('T')) / n
    return [len(s), kd, ch, ar, gly, ser]


class LiftTable:
    """Shrunk log-lift  log[ P(b|a) / P(b) ]  with Dirichlet-on-marginal smoothing."""

    def __init__(self, alpha=30.0):
        self.alpha = alpha

    def fit(self, a, b):
        self.cab = collections.Counter(zip(a, b))
        self.ca = collections.Counter(a)
        self.cb = collections.Counter(b)
        self.N = len(a)
        return self

    def __call__(self, a, b):
        pb = self.cb.get(b, 0) / self.N
        if pb <= 0:
            return 0.0
        num = self.cab.get((a, b), 0) + self.alpha * pb
        den = self.ca.get(a, 0) + self.alpha
        return float(np.log((num / den) / pb))

    def vec(self, A, B):
        return np.array([self(x, y) for x, y in zip(A, B)])

    def loo(self, a, b_true, b):
        """Lift for (a,b) with the single observation (a,b_true) removed from the table.

        Training rows contributed their own true pair to these counts. Scoring them without
        this subtraction leaks the label into the feature: the true candidate's lift is
        inflated by exactly its own count, the decoys' is not. The model then leans on a
        feature that is weaker at validation time than it looked at training time.
        """
        N = self.N - 1
        if N <= 0:
            return 0.0
        cb = self.cb.get(b, 0) - (1 if b == b_true else 0)
        pb = cb / N
        if pb <= 0:
            return 0.0
        num = self.cab.get((a, b), 0) - (1 if b == b_true else 0) + self.alpha * pb
        den = self.ca.get(a, 0) - 1 + self.alpha
        if den <= 0 or num <= 0:
            return 0.0
        return float(np.log((num / den) / pb))


class FeatureBuilder:
    def __init__(self, shm_w=110):
        self.shm_w = shm_w

    # ---------- fit on a training slice ----------
    def fit(self, D, idx):
        df = D['df']; lv = D['lv']; lc = D['lc']; aas = D['aas']; y = D['y']
        sub = df.iloc[idx]
        tl_v = lv[idx, y[idx]]; tl_j = D['lj'][idx, y[idx]]
        tl_c = lc[idx, y[idx]]; tl_a = aas[idx, y[idx]]
        self.hcons = BackConsensus(self.shm_w).fit(
            sub['heavy_v_gene'].values, sub['heavy_chain_aa'].values, sub['heavy_cdr3_aa'].values)
        self.lcons = BackConsensus(self.shm_w).fit(tl_v, tl_a, tl_c)

        hv = sub['heavy_v_gene'].values; hj = sub['heavy_j_gene'].values
        hd = sub['heavy_d_gene'].values
        hfam = np.array([g.split('-')[0] for g in hv])
        lfam = np.array([g.split('-')[0] for g in tl_v])
        loc = np.array([g[:3] for g in tl_v])
        self.t_vv = LiftTable(30).fit(hv, tl_v)
        self.t_vfam = LiftTable(20).fit(hv, lfam)
        self.t_famv = LiftTable(20).fit(hfam, tl_v)
        self.t_famfam = LiftTable(10).fit(hfam, lfam)
        self.t_vloc = LiftTable(20).fit(hv, loc)
        self.t_famloc = LiftTable(10).fit(hfam, loc)
        self.t_jj = LiftTable(20).fit(hj, tl_j)
        self.t_jv = LiftTable(30).fit(hj, tl_v)
        self.t_vj = LiftTable(30).fit(hv, tl_j)
        self.t_jloc = LiftTable(10).fit(hj, loc)
        self.t_dv = LiftTable(30).fit(hd, lfam)
        # heavy cdr3 length bucket x light V
        hb = np.clip(sub['heavy_cdr3_aa'].str.len().values // 3, 0, 9)
        self.t_lenv = LiftTable(30).fit(hb, tl_v)
        self.t_lenloc = LiftTable(10).fit(hb, loc)
        # vocab
        self.vocab = {}
        for name, vals in (('hv', hv), ('hj', hj), ('hd', hd), ('lv', tl_v), ('lj', tl_j)):
            u = sorted(set(vals))
            self.vocab[name] = {v: i + 1 for i, v in enumerate(u)}  # 0 = OOV
        self.vocab['sp'] = {v: i + 1 for i, v in enumerate(sorted(set(sub['species'].values)))}
        return self

    # ---------- apply ----------
    def transform(self, D, idx):
        df = D['df'].iloc[idx]; n = len(idx)
        lv = D['lv'][idx]; lj = D['lj'][idx]; lc = D['lc'][idx]; aas = D['aas'][idx]
        hv = df['heavy_v_gene'].values; hj = df['heavy_j_gene'].values
        hd = df['heavy_d_gene'].values; hc3 = df['heavy_cdr3_aa'].values
        haa = df['heavy_chain_aa'].values; blk = df['block_id'].values
        hfam = np.array([g.split('-')[0] for g in hv])

        h_rate, h_cnt, h_cov = self.hcons.mut(hv, haa, hc3)
        fl_v = lv.reshape(-1); fl_a = aas.reshape(-1); fl_c = lc.reshape(-1)
        l_rate, l_cnt, l_cov = self.lcons.mut(fl_v, fl_a, fl_c)
        l_rate = l_rate.reshape(n, 8); l_cnt = l_cnt.reshape(n, 8); l_cov = l_cov.reshape(n, 8)

        # within-block ranks
        hr_rank = pd.Series(h_rate).groupby(pd.Series(blk)).rank(method='average').values - 1.0
        hc_rank = pd.Series(h_cnt).groupby(pd.Series(blk)).rank(method='average').values - 1.0
        lr_rank = np.argsort(np.argsort(l_rate, axis=1, kind='stable'), axis=1).astype(float)
        lc_rank = np.argsort(np.argsort(l_cnt, axis=1, kind='stable'), axis=1).astype(float)
        # block-standardised values
        hz = pd.Series(h_rate).groupby(pd.Series(blk)).transform(lambda v: (v - v.mean()) / (v.std() + 1e-6)).values
        lz = (l_rate - l_rate.mean(1, keepdims=True)) / (l_rate.std(1, keepdims=True) + 1e-6)
        hzc = pd.Series(h_cnt).groupby(pd.Series(blk)).transform(lambda v: (v - v.mean()) / (v.std() + 1e-6)).values
        lzc = (l_cnt - l_cnt.mean(1, keepdims=True)) / (l_cnt.std(1, keepdims=True) + 1e-6)

        lfam = np.array([[g.split('-')[0] for g in row] for row in lv], dtype=object)
        loc = np.array([[g[:3] for g in row] for row in lv], dtype=object)
        hb = np.clip(pd.Series(hc3).str.len().values // 3, 0, 9)

        hprops = np.array([seq_props(s) for s in hc3])
        lprops = np.array([seq_props(s) for s in fl_c]).reshape(n, 8, -1)
        hprops_full = np.array([seq_props(s) for s in haa])
        lprops_full = np.array([seq_props(s) for s in fl_a]).reshape(n, 8, -1)

        # pair features (n,8,F)
        F = []
        rep = lambda v: np.repeat(v[:, None], 8, axis=1)
        for t, A, B in ((self.t_vv, hv, lv), (self.t_vfam, hv, lfam), (self.t_famv, hfam, lv),
                        (self.t_famfam, hfam, lfam), (self.t_vloc, hv, loc), (self.t_famloc, hfam, loc),
                        (self.t_jj, hj, lj), (self.t_jv, hj, lv), (self.t_vj, hv, lj),
                        (self.t_jloc, hj, loc), (self.t_dv, hd, lfam),
                        (self.t_lenv, hb, lv), (self.t_lenloc, hb, loc)):
            F.append(np.array([[t(A[i], B[i][j]) for j in range(8)] for i in range(n)]))
        F.append(-np.abs(hr_rank[:, None] - lr_rank))
        F.append(-np.abs(hc_rank[:, None] - lc_rank))
        F.append(-np.abs(hz[:, None] - lz))
        F.append(-np.abs(hzc[:, None] - lzc))
        F.append(hz[:, None] * lz)
        F.append(-np.abs(rep(h_rate) - l_rate))
        F.append(-np.abs(rep(h_cnt) - l_cnt))
        F.append(rep(h_rate) - l_rate)
        F.append(rep(h_rate) * l_rate)
        F.append(l_rate); F.append(lz); F.append(lr_rank)
        F.append(rep(h_rate)); F.append(rep(hz)); F.append(rep(hr_rank))
        F.append(l_cov / 100.0)
        # cdr3 property interactions
        for k in range(hprops.shape[1]):
            F.append(rep(hprops[:, k]) * 0 + lprops[:, :, k])
            F.append(-np.abs(rep(hprops[:, k]) - lprops[:, :, k]))
            F.append(rep(hprops[:, k]) * lprops[:, :, k])
        for k in (0, 1, 2):
            F.append(-np.abs(rep(hprops_full[:, k]) - lprops_full[:, :, k]))
            F.append(lprops_full[:, :, k])
        X = np.stack(F, axis=-1).astype(np.float32)

        # categorical ids
        V = self.vocab
        cat_h = np.stack([
            np.array([V['hv'].get(x, 0) for x in hv]),
            np.array([V['hj'].get(x, 0) for x in hj]),
            np.array([V['hd'].get(x, 0) for x in hd]),
            np.array([V['sp'].get(x, 0) for x in df['species'].values]),
        ], axis=1)
        cat_l = np.stack([
            np.array([[V['lv'].get(x, 0) for x in row] for row in lv]),
            np.array([[V['lj'].get(x, 0) for x in row] for row in lj]),
        ], axis=-1)
        return dict(X=X, cat_h=cat_h.astype(np.int64), cat_l=cat_l.astype(np.int64),
                    block=blk, h_rate=h_rate, l_rate=l_rate)

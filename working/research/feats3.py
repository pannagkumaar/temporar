"""v3 features.

Decisions baked in, each with its measured reason:
  * hamming-to-modal-consensus, NOT surprisal (0.364 vs 0.311 alone).
  * one-pass consensus; two-pass made no difference at any keep fraction (0.3639 either way).
  * both the raw clock and the per-gene-percentile clock (0.3737 together vs 0.3639 / 0.3556 alone).
  * leave-one-out lift tables on the training slice (each row contributed its own true pair).
  * no per-position residue towers: train 0.76 / val 0.45, pure memorisation.
"""
import numpy as np, pandas as pd, collections
from shm import BackConsensus
from feats import LiftTable, seq_props

W = 110
WINDOWS = ((0, 110), (0, 40), (40, 75), (75, 110))
LIFTS = (('vv', 30), ('vfam', 20), ('famv', 20), ('famfam', 10), ('vloc', 20), ('famloc', 10),
         ('jj', 20), ('jv', 30), ('vj', 30), ('jloc', 10), ('dv', 30), ('lenv', 30), ('lenloc', 10))


def _rank(v, key):
    return pd.Series(v).groupby(pd.Series(key)).rank(method='average').values - 1.0


def _pct(v, key):
    return pd.Series(v).groupby(pd.Series(key)).rank(pct=True).values


def _z(v, key):
    return pd.Series(v).groupby(pd.Series(key)).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)).values


def _crank(l):
    return np.argsort(np.argsort(l, 1, kind='stable'), 1).astype(np.float32)


def _cz(l):
    return (l - l.mean(1, keepdims=True)) / (l.std(1, keepdims=True) + 1e-6)


def seqsim(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


class FB3:
    def fit(self, D, idx):
        df = D['df'].iloc[idx]
        lv = D['lv'][idx]; lj = D['lj'][idx]; lc = D['lc'][idx]; aas = D['aas'][idx]
        y = D['y'][idx]; r = np.arange(len(idx))
        tv = lv[r, y]; tj = lj[r, y]; tc = lc[r, y]; ta = aas[r, y]
        hv = df['heavy_v_gene'].values; hj = df['heavy_j_gene'].values; hd = df['heavy_d_gene'].values
        haa = df['heavy_chain_aa'].values; hc3 = df['heavy_cdr3_aa'].values

        self.hcons = {}; self.lcons = {}
        for (a, b) in WINDOWS:
            self.hcons[(a, b)] = BackConsensus(b).fit(hv, haa, hc3)
            self.lcons[(a, b)] = BackConsensus(b).fit(tv, ta, tc)
        # per-gene reference distribution of the full-window clock (for the percentile view)
        hr, _, _ = self.hcons[WINDOWS[0]].mut(hv, haa, hc3)
        lr, _, _ = self.lcons[WINDOWS[0]].mut(tv, ta, tc)
        self.href = {g: np.sort(hr[hv == g]) for g in sorted(set(hv))}
        self.lref = {g: np.sort(lr[tv == g]) for g in sorted(set(tv))}
        self.hall = np.sort(hr); self.lall = np.sort(lr)

        hfam = np.array([g.split('-')[0] for g in hv])
        lfam = np.array([g.split('-')[0] for g in tv]); loc = np.array([g[:3] for g in tv])
        hb = np.clip(df['heavy_cdr3_aa'].str.len().values // 3, 0, 9)
        src = dict(vv=(hv, tv), vfam=(hv, lfam), famv=(hfam, tv), famfam=(hfam, lfam),
                   vloc=(hv, loc), famloc=(hfam, loc), jj=(hj, tj), jv=(hj, tv), vj=(hv, tj),
                   jloc=(hj, loc), dv=(hd, lfam), lenv=(hb, tv), lenloc=(hb, loc))
        self.T = {k: LiftTable(al).fit(*src[k]) for k, al in LIFTS}
        self.vocab = {}
        for nm, vals in (('hv', hv), ('hj', hj), ('hd', hd), ('lv', tv), ('lj', tj)):
            self.vocab[nm] = {v: i + 1 for i, v in enumerate(sorted(set(vals)))}
        self.vocab['sp'] = {v: i + 1 for i, v in enumerate(sorted(set(df['species'].values)))}
        return self

    @staticmethod
    def _pct_ref(ref_map, allref, genes, vals):
        out = np.empty(len(vals), dtype=np.float32)
        for i, (g, v) in enumerate(zip(genes, vals)):
            ref = ref_map.get(g)
            if ref is None or len(ref) < 8:
                ref = allref
            out[i] = np.searchsorted(ref, v, 'left') / max(len(ref), 1)
        return out

    def transform(self, D, idx, loo=False):
        df = D['df'].iloc[idx]; n = len(idx)
        lv = D['lv'][idx]; lj = D['lj'][idx]; lc = D['lc'][idx]; aas = D['aas'][idx]
        hv = df['heavy_v_gene'].values; hj = df['heavy_j_gene'].values
        hd = df['heavy_d_gene'].values; hc3 = df['heavy_cdr3_aa'].values
        haa = df['heavy_chain_aa'].values; blk = df['block_id'].values
        hfam = np.array([g.split('-')[0] for g in hv])
        flv = lv.reshape(-1); fla = aas.reshape(-1); flc = lc.reshape(-1)
        rep = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], 8, 1)
        F = []

        for wi, wk in enumerate(WINDOWS):
            a, b = wk
            hrate, hcnt, hcov = self.hcons[wk].mut(hv, haa, hc3)
            lrate, lcnt, lcov = self.lcons[wk].mut(flv, fla, flc)
            lrate = lrate.reshape(n, 8); lcnt = lcnt.reshape(n, 8); lcov = lcov.reshape(n, 8)
            for h, l in ((hrate, lrate), (hcnt, lcnt)):
                hrk = _rank(h, blk); lrk = _crank(l)
                hzz = _z(h, blk); lzz = _cz(l)
                F += [-np.abs(hrk[:, None] - lrk), -np.abs(hzz[:, None] - lzz), hzz[:, None] * lzz,
                      -np.abs(rep(h) - l), rep(h) - l, l, lzz, rep(h), rep(hzz)]
            if wi == 0:
                F.append(lcov / 100.0)
                # per-gene percentile clock (complementary view, +0.010 alone)
                hp = self._pct_ref(self.href, self.hall, hv, hrate)
                lp = self._pct_ref(self.lref, self.lall, flv, lrate.reshape(-1)).reshape(n, 8)
                hprk = _rank(hp, blk); lprk = _crank(lp)
                hpz = _z(hp, blk); lpz = _cz(lp)
                F += [-np.abs(hprk[:, None] - lprk), -np.abs(hpz[:, None] - lpz),
                      hpz[:, None] * lpz, -np.abs(rep(hp) - lp), rep(hp) - lp, lp, lpz,
                      rep(hp), rep(hpz)]

        lfam = np.array([[g.split('-')[0] for g in row] for row in lv], dtype=object)
        loc = np.array([[g[:3] for g in row] for row in lv], dtype=object)
        hb = np.clip(pd.Series(hc3).str.len().values // 3, 0, 9)
        B = dict(vv=lv, vfam=lfam, famv=lv, famfam=lfam, vloc=loc, famloc=loc, jj=lj, jv=lv,
                 vj=lj, jloc=loc, dv=lfam, lenv=lv, lenloc=loc)
        A = dict(vv=hv, vfam=hv, famv=hfam, famfam=hfam, vloc=hv, famloc=hfam, jj=hj, jv=hj,
                 vj=hv, jloc=hj, dv=hd, lenv=hb, lenloc=hb)
        yy = D['y'][idx] if loo else None
        for key, _ in LIFTS:
            t = self.T[key]; Ak = A[key]; Bk = B[key]
            if loo:
                F.append(np.array([[t.loo(Ak[i], Bk[i][yy[i]], Bk[i][j]) for j in range(8)]
                                   for i in range(n)], np.float32))
            else:
                F.append(np.array([[t(Ak[i], Bk[i][j]) for j in range(8)] for i in range(n)],
                                  np.float32))

        hp_ = np.array([seq_props(s) for s in hc3])
        lp_ = np.array([seq_props(s) for s in flc]).reshape(n, 8, -1)
        for k in range(hp_.shape[1]):
            F += [lp_[:, :, k], -np.abs(rep(hp_[:, k]) - lp_[:, :, k]), rep(hp_[:, k]) * lp_[:, :, k]]

        F += self._clonal(blk, hc3, hv, lc, lv)
        X = np.stack(F, -1).astype(np.float32)

        V = self.vocab
        cat_h = np.stack([np.array([V['hv'].get(x, 0) for x in hv]),
                          np.array([V['hj'].get(x, 0) for x in hj]),
                          np.array([V['hd'].get(x, 0) for x in hd]),
                          np.array([V['sp'].get(x, 0) for x in df['species'].values])], 1)
        cat_l = np.stack([np.array([[V['lv'].get(x, 0) for x in row] for row in lv]),
                          np.array([[V['lj'].get(x, 0) for x in row] for row in lj])], -1)
        return dict(X=X, cat_h=cat_h.astype(np.int64), cat_l=cat_l.astype(np.int64), block=blk)

    @staticmethod
    def _clonal(blk, hc3, hv, lc, lv):
        """Within-block relational agreement. Clonal relatives are rare (~0.04% of within-block
        pairs) but when two heavies are clonal their light chains are near-identical, which
        constrains the assignment. Assignment-free proxy: compare each heavy's sorted
        neighbour-similarity profile to each candidate's."""
        n = len(blk)
        f_dot = np.zeros((n, 8), np.float32); f_max = np.zeros((n, 8), np.float32)
        order = pd.Series(np.arange(n), index=blk)
        for _, g in order.groupby(level=0, sort=False):
            ii = g.values
            m = len(ii)
            Hs = np.zeros((m, m), np.float32)
            for a in range(m):
                for b in range(a + 1, m):
                    s = seqsim(hc3[ii[a]], hc3[ii[b]]) * (hv[ii[a]] == hv[ii[b]])
                    Hs[a, b] = Hs[b, a] = s
            j0 = ii[0]
            Ls = np.zeros((8, 8), np.float32)
            for a in range(8):
                for b in range(a + 1, 8):
                    s = seqsim(lc[j0, a], lc[j0, b]) * (lv[j0, a] == lv[j0, b])
                    Ls[a, b] = Ls[b, a] = s
            hs = -np.sort(-Hs, 1)[:, :3]; ls = -np.sort(-Ls, 1)[:, :3]
            f_dot[ii] = hs @ ls.T
            f_max[ii] = -np.abs(hs[:, :1] - ls[:, 0][None, :])
        return [f_dot, f_max]

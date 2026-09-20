"""v2 features: probabilistic windowed SHM + gene lifts + CDR3-anchored residue identities."""
import numpy as np, pandas as pd, collections
from shm2 import ProbGermline
from feats import LiftTable, seq_props

NRES = 24          # backwards positions kept as residue identities
WIN = ((0, 16), (16, 32), (32, 48), (48, 72), (72, 112), (0, 112))
NW = len(WIN)


def _blockrank(v, blk):
    s = pd.Series(v)
    return s.groupby(pd.Series(blk)).rank(method='average').values - 1.0


def _blockz(v, blk):
    s = pd.Series(v)
    return s.groupby(pd.Series(blk)).transform(lambda x: (x - x.mean()) / (x.std() + 1e-6)).values


class FB2:
    def __init__(self, w=112, use_res=True):
        self.w = w; self.use_res = use_res

    def fit(self, D, idx):
        df = D['df'].iloc[idx]
        lv = D['lv'][idx]; lj = D['lj'][idx]; lc = D['lc'][idx]; aas = D['aas'][idx]
        y = D['y'][idx]; r = np.arange(len(idx))
        tv = lv[r, y]; tj = lj[r, y]; tc = lc[r, y]; ta = aas[r, y]

        self.gh = ProbGermline(self.w, windows=WIN).fit(
            df['heavy_v_gene'].values, df['heavy_chain_aa'].values, df['heavy_cdr3_aa'].values)
        self.gl = ProbGermline(self.w, windows=WIN).fit(tv, ta, tc)

        hv = df['heavy_v_gene'].values; hj = df['heavy_j_gene'].values; hd = df['heavy_d_gene'].values
        hfam = np.array([g.split('-')[0] for g in hv])
        lfam = np.array([g.split('-')[0] for g in tv]); loc = np.array([g[:3] for g in tv])
        hb = np.clip(df['heavy_cdr3_aa'].str.len().values // 3, 0, 9)
        self.T = {
            'vv': LiftTable(30).fit(hv, tv), 'vfam': LiftTable(20).fit(hv, lfam),
            'famv': LiftTable(20).fit(hfam, tv), 'famfam': LiftTable(10).fit(hfam, lfam),
            'vloc': LiftTable(20).fit(hv, loc), 'famloc': LiftTable(10).fit(hfam, loc),
            'jj': LiftTable(20).fit(hj, tj), 'jv': LiftTable(30).fit(hj, tv),
            'vj': LiftTable(30).fit(hv, tj), 'jloc': LiftTable(10).fit(hj, loc),
            'dv': LiftTable(30).fit(hd, lfam), 'lenv': LiftTable(30).fit(hb, tv),
            'lenloc': LiftTable(10).fit(hb, loc), 'vlj': LiftTable(30).fit(hv, tj),
        }
        self.vocab = {}
        for name, vals in (('hv', hv), ('hj', hj), ('hd', hd), ('lv', tv), ('lj', tj)):
            self.vocab[name] = {v: i + 1 for i, v in enumerate(sorted(set(vals)))}
        self.vocab['sp'] = {v: i + 1 for i, v in enumerate(sorted(set(df['species'].values)))}
        return self

    def transform(self, D, idx, loo=False):
        """loo=True for rows that are inside the fitted tables (i.e. the training slice):
        each row's own true pair is subtracted from every lift count before scoring it."""
        df = D['df'].iloc[idx]; n = len(idx)
        lv = D['lv'][idx]; lj = D['lj'][idx]; lc = D['lc'][idx]; aas = D['aas'][idx]
        hv = df['heavy_v_gene'].values; hj = df['heavy_j_gene'].values
        hd = df['heavy_d_gene'].values; hc3 = df['heavy_cdr3_aa'].values
        haa = df['heavy_chain_aa'].values; blk = df['block_id'].values
        hfam = np.array([g.split('-')[0] for g in hv])
        flv = lv.reshape(-1); fla = aas.reshape(-1); flc = lc.reshape(-1)

        H = self.gh.score(hv, haa, hc3)                       # (n, NW*3)
        L = self.gl.score(flv, fla, flc).reshape(n, 8, NW * 3)

        F = []
        rep = lambda v: np.repeat(np.asarray(v, dtype=np.float32)[:, None], 8, axis=1)
        for k in range(NW):
            for off, scale in ((0, 1.0), (1, 1.0)):            # surprisal, expected-mismatch
                h = H[:, k * 3 + off] / np.maximum(H[:, k * 3 + 2], 1.0)
                l = L[:, :, k * 3 + off] / np.maximum(L[:, :, k * 3 + 2], 1.0)
                hr = _blockrank(h, blk); lr = np.argsort(np.argsort(l, 1, kind='stable'), 1).astype(float)
                hz = _blockz(h, blk)
                lz = (l - l.mean(1, keepdims=True)) / (l.std(1, keepdims=True) + 1e-6)
                F += [-np.abs(hr[:, None] - lr), -np.abs(hz[:, None] - lz), hz[:, None] * lz,
                      -np.abs(rep(h) - l), rep(h) - l, l, lz, rep(h), rep(hz)]
            # absolute counts too (not rate)
            hcn = H[:, k * 3 + 1]; lcn = L[:, :, k * 3 + 1]
            F += [-np.abs(rep(hcn) - lcn), lcn, rep(hcn)]
        F.append(L[:, :, 2] / 100.0)

        lfam = np.array([[g.split('-')[0] for g in row] for row in lv], dtype=object)
        loc = np.array([[g[:3] for g in row] for row in lv], dtype=object)
        hb = np.clip(pd.Series(hc3).str.len().values // 3, 0, 9)
        yy = D['y'][idx] if loo else None
        r8 = range(8)
        for key, A, B in (('vv', hv, lv), ('vfam', hv, lfam), ('famv', hfam, lv),
                          ('famfam', hfam, lfam), ('vloc', hv, loc), ('famloc', hfam, loc),
                          ('jj', hj, lj), ('jv', hj, lv), ('vj', hv, lj), ('jloc', hj, loc),
                          ('dv', hd, lfam), ('lenv', hb, lv), ('lenloc', hb, loc)):
            t = self.T[key]
            if loo:
                F.append(np.array([[t.loo(A[i], B[i][yy[i]], B[i][j]) for j in r8]
                                   for i in range(n)]))
            else:
                F.append(np.array([[t(A[i], B[i][j]) for j in r8] for i in range(n)]))

        hp = np.array([seq_props(s) for s in hc3])
        lp = np.array([seq_props(s) for s in flc]).reshape(n, 8, -1)
        for k in range(hp.shape[1]):
            F += [lp[:, :, k], -np.abs(rep(hp[:, k]) - lp[:, :, k]), rep(hp[:, k]) * lp[:, :, k]]
        X = np.stack(F, axis=-1).astype(np.float32)

        V = self.vocab
        cat_h = np.stack([np.array([V['hv'].get(x, 0) for x in hv]),
                          np.array([V['hj'].get(x, 0) for x in hj]),
                          np.array([V['hd'].get(x, 0) for x in hd]),
                          np.array([V['sp'].get(x, 0) for x in df['species'].values])], 1)
        cat_l = np.stack([np.array([[V['lv'].get(x, 0) for x in row] for row in lv]),
                          np.array([[V['lj'].get(x, 0) for x in row] for row in lj])], -1)
        out = dict(X=X, cat_h=cat_h.astype(np.int64), cat_l=cat_l.astype(np.int64), block=blk)
        if self.use_res:
            out['res_h'] = self.gh.residues(haa, hc3, NRES)                       # (n,NRES)
            out['res_l'] = self.gl.residues(fla, flc, NRES).reshape(n, 8, NRES)   # (n,8,NRES)
        return out

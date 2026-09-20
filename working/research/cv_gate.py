"""Local CPU test of the block-gate hypothesis (v3 vs v4 features).

Runs against the mean-pooled embeddings already on disk rather than the mean+CDR3 pooling the
delivery uses, so the absolute level is lower than the incumbent. That is fine: the question is
whether the 12 block-dispersion gate features help, and both arms share the same embeddings,
folds and seeds.
"""
import sys, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from feats3 import FB3
from feats4 import FB4
from model5 import BlockMatcher5
from model import block_order
from permmarg import marginals
from cv3 import make_folds

torch.set_num_threads(8)
ART = '.mlctl/artifacts/abpair-emb3'
NAMES = ['antiberta2', 'igbert', 'esm2']


def zs(A, rows):
    A = A.astype(np.float32); s = A[rows]
    return (A - s.mean(0, keepdims=True)) / (s.std(0, keepdims=True) + 1e-5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    D = load('train.csv'); df = D['df']; y = D['y']
    fold = make_folds(df, 5); hum = df['species'].values == 'human'
    heavy = open(f'{ART}/heavy_seqs.txt').read().split('\n')
    light = open(f'{ART}/light_seqs.txt').read().split('\n')
    hi = {s: i for i, s in enumerate(heavy)}; li = {s: i for i, s in enumerate(light)}
    IDX_H = np.array([hi[s] for s in df['heavy_chain_aa'].astype(str)], np.int64)
    IDX_L = np.array([[li[s] for s in row] for row in D['aas']], np.int64)
    EH_raw = [np.load(f'{ART}/{n}_heavy.npy') for n in NAMES]
    EL_raw = [np.load(f'{ART}/{n}_light.npy') for n in NAMES]

    def fit(Ftr, ytr, Fva, yva, voc, EH, EL, ihtr, iltr, ihva, ilva, seed, epochs=14):
        torch.manual_seed(seed); np.random.seed(seed)
        otr, Btr = block_order(Ftr['block']); ova, _ = block_order(Fva['block'])
        T = lambda x, o: torch.tensor(x[o])
        Xt, Ht, Lt, Yt = T(Ftr['X'], otr), T(Ftr['cat_h'], otr), T(Ftr['cat_l'], otr), T(ytr, otr)
        Xv, Hv, Lv = T(Fva['X'], ova), T(Fva['cat_h'], ova), T(Fva['cat_l'], ova)
        IHt, ILt, IHv, ILv = T(ihtr, otr), T(iltr, otr), T(ihva, ova), T(ilva, ova)
        m = BlockMatcher5(Xt.shape[-1], voc, EH.shape[1])
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
        nb = 96; steps = epochs * ((Btr + nb - 1) // nb)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.25)
        g = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            m.train(); perm = torch.randperm(Btr, generator=g)
            for k in range(0, Btr, nb):
                ri = (perm[k:k + nb][:, None] * 8 + torch.arange(8)).reshape(-1)
                M, _ = m(Xt[ri], Ht[ri], Lt[ri], IHt[ri], ILt[ri], EH, EL, sinkhorn_iter=6)
                tg = Yt[ri].reshape(-1, 8)
                loss = Fn.cross_entropy(M.reshape(-1, 8), tg.reshape(-1)) + 0.5 * Fn.cross_entropy(
                    M.transpose(1, 2).reshape(-1, 8), torch.argsort(tg, 1).reshape(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            Mr, _ = m(Xv, Hv, Lv, IHv, ILv, EH, EL, sinkhorn_iter=30)
        return Mr.reshape(-1, 8).numpy()[np.argsort(ova)]

    for tag, FB in (('v3', FB3), ('v4_blockgate', FB4)):
        t0 = time.time(); raw = np.zeros((len(df), 8), np.float32); pf = []
        for f in range(5):
            tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
            fb = FB().fit(D, tri)
            Ftr = fb.transform(D, tri, loo=True); Fva = fb.transform(D, vai)
            voc = {k: len(v) for k, v in fb.vocab.items()}
            rh = np.unique(IDX_H[tri]); rl = np.unique(IDX_L[tri].reshape(-1))
            EH = torch.tensor(np.concatenate([zs(A, rh) for A in EH_raw], 1))
            EL = torch.tensor(np.concatenate([zs(A, rl) for A in EL_raw], 1))
            acc = np.zeros((len(vai), 8))
            for s in range(a.seeds):
                acc += fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL, IDX_H[tri], IDX_L[tri],
                           IDX_H[vai], IDX_L[vai], seed=s)
            raw[vai] = acc / a.seeds
            o, B = block_order(Fva['block']); inv = np.argsort(o)
            p = marginals(raw[vai][o].reshape(B, 8, 8), 1.25)
            pf.append(adjusted(credit_from_scores(np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv], y[vai])))
            print(f'  [{tag}] fold {f} adj={pf[-1]:.4f} nfeat={Ftr["X"].shape[-1]} '
                  f'[{time.time()-t0:.0f}s]', flush=True)
            del EH, EL
        o, B = block_order(df['block_id'].values); inv = np.argsort(o)
        p = marginals(raw[o].reshape(B, 8, 8), 1.25)
        sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
        print(f'== {tag}: OOF={adjusted(credit_from_scores(sc, y)):.4f} '
              f'human={adjusted(credit_from_scores(sc[hum], y[hum])):.4f} '
              f'folds={[round(x,4) for x in pf]}', flush=True)
        np.save(f'working/cache/oofraw_local_{tag}.npy', raw)


main()

"""Decisive comparison: do frozen antibody-LM embeddings add signal over the engineered features?
Same folds, same features, same seeds; only the LM towers change."""
import sys, os, time, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from feats3 import FB3
from model4 import BlockMatcher4
from model import block_order
from permmarg import marginals
from cv3 import make_folds

torch.set_num_threads(8)
CACHE = 'working/cache'
ART = '.mlctl/artifacts/abpair-emb3'


def _zs(A):
    return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-5)


def load_emb(name, D):
    """Return (EH, EL, idx_h, idx_l) aligned to the loaded dataframe rows.
    `name` may be comma-separated; each model is z-scored then concatenated."""
    heavy = open(f'{ART}/heavy_seqs.txt').read().split('\n')
    light = open(f'{ART}/light_seqs.txt').read().split('\n')
    names = [x for x in name.split(',') if x]
    EH = np.concatenate([_zs(np.load(f'{ART}/{m}_heavy.npy').astype(np.float32)) for m in names], 1)
    EL = np.concatenate([_zs(np.load(f'{ART}/{m}_light.npy').astype(np.float32)) for m in names], 1)
    hi = {s: i for i, s in enumerate(heavy)}
    li = {s: i for i, s in enumerate(light)}
    df = D['df']
    idx_h = np.array([hi[s] for s in df['heavy_chain_aa'].astype(str)], np.int64)
    idx_l = np.array([[li[s] for s in row] for row in D['aas']], np.int64)
    return EH, EL, idx_h, idx_l


def fit_fold(Ftr, ytr, Fva, yva, vocab, EH=None, EL=None, seed=0, epochs=14, lr=3e-3,
             wd=1e-4, sink_train=6, hid=192, d=48, pdrop=0.15, colw=0.5, nb=96,
             de=64, edrop=0.3, dev='cpu'):
    torch.manual_seed(seed); np.random.seed(seed)
    otr, Btr = block_order(Ftr['block']); ova, _ = block_order(Fva['block'])

    def pk(F, y, o):
        t = [torch.tensor(F['X'][o]), torch.tensor(F['cat_h'][o]), torch.tensor(F['cat_l'][o]),
             torch.tensor(y[o])]
        t += [torch.tensor(F['ih'][o]), torch.tensor(F['il'][o])] if 'ih' in F else [None, None]
        return t
    Xt, Ht, Lt, Yt, IHt, ILt = pk(Ftr, ytr, otr)
    Xv, Hv, Lv, Yv, IHv, ILv = pk(Fva, yva, ova)
    m = BlockMatcher4(Xt.shape[-1], vocab, EH, EL, d=d, hid=hid, pdrop=pdrop, de=de, edrop=edrop)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((Btr + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)
    sl = lambda t, i: (None if t is None else t[i])
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, nb):
            ri = (perm[k:k + nb][:, None] * 8 + torch.arange(8)).reshape(-1)
            M, _ = m(Xt[ri], Ht[ri], Lt[ri], sl(IHt, ri), sl(ILt, ri), sinkhorn_iter=sink_train)
            tgt = Yt[ri].reshape(-1, 8)
            loss = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1)) + colw * Fn.cross_entropy(
                M.transpose(1, 2).reshape(-1, 8), torch.argsort(tgt, 1).reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Mr, _ = m(Xv, Hv, Lv, IHv, ILv, sinkhorn_iter=30)
    inv = np.argsort(ova)
    return Mr.reshape(-1, 8).cpu().numpy()[inv]


def run(tag, emb=None, seeds=(0, 1, 2), nf=5, loo=True, T=1.25, **kw):
    t0 = time.time()
    D = load('train.csv'); df = D['df']; y = D['y']
    fold = make_folds(df, nf)
    EH = EL = None; ih = il = None
    if emb:
        EH, EL, ih, il = load_emb(emb, D)
        print(f'  emb {emb}: EH{EH.shape} EL{EL.shape}', flush=True)
    oofraw = np.zeros((len(df), 8), np.float32); pf = []
    for f in range(nf):
        tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
        cp = f'{CACHE}/f3_fold{f}_loo{int(loo)}.npz'
        z = np.load(cp, allow_pickle=True)
        Ftr = {k: z['tr_' + k] for k in ('X', 'cat_h', 'cat_l', 'block')}
        Fva = {k: z['va_' + k] for k in ('X', 'cat_h', 'cat_l', 'block')}
        voc = z['voc'].item()
        if emb:
            Ftr['ih'] = ih[tri]; Ftr['il'] = il[tri]
            Fva['ih'] = ih[vai]; Fva['il'] = il[vai]
        acc = np.zeros((len(vai), 8))
        for s in seeds:
            acc += fit_fold(Ftr, y[tri], Fva, y[vai], voc, EH, EL, seed=s, **kw)
        oofraw[vai] = acc / len(seeds)
        o, B = block_order(Fva['block']); inv = np.argsort(o)
        p = marginals(oofraw[vai][o].reshape(B, 8, 8), T)
        sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
        a = adjusted(credit_from_scores(sc, y[vai])); pf.append(a)
        print(f'  [{tag}] fold {f} adj={a:.4f} [{time.time()-t0:.0f}s]', flush=True)
    o, B = block_order(df['block_id'].values); inv = np.argsort(o)
    p = marginals(oofraw[o].reshape(B, 8, 8), T)
    sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
    c = credit_from_scores(sc, y); hum = df['species'].values == 'human'
    print(f'== {tag}: OOF={adjusted(c):.4f} human={adjusted(c[hum]):.4f} minfold={min(pf):.4f} '
          f'proxy={0.75*adjusted(c)+0.25*min(pf):.4f}', flush=True)
    np.save(f'{CACHE}/oofraw_{tag}.npy', oofraw)
    return adjusted(c)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', default=''); ap.add_argument('--tag', default='')
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--de', type=int, default=64); ap.add_argument('--edrop', type=float, default=0.3)
    a = ap.parse_args()
    run(a.tag or (a.emb or 'base'), emb=(a.emb or None), seeds=tuple(range(a.seeds)),
        epochs=a.epochs, de=a.de, edrop=a.edrop)

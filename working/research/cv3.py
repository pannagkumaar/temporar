import sys, os, time, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from feats3 import FB3
from model import BlockMatcher, block_order

torch.set_num_threads(8)
CACHE = 'working/cache'


def make_folds(df, nf=5, seed=7):
    s = df['sample_id'].values
    uq = np.array(sorted(set(s)))
    perm = np.random.default_rng(seed).permutation(len(uq))
    fo = {g: perm[i] % nf for i, g in enumerate(uq)}
    return np.array([fo[g] for g in s])


def fit_fold(Ftr, ytr, Fva, yva, vocab, seed=0, epochs=14, lr=3e-3, wd=1e-4, sink_train=6,
             hid=192, d=48, pdrop=0.15, colw=0.5, nb=96, dev='cpu', ret_train=False):
    torch.manual_seed(seed); np.random.seed(seed)
    otr, Btr = block_order(Ftr['block']); ova, _ = block_order(Fva['block'])
    pk = lambda F, y, o: (torch.tensor(F['X'][o], device=dev), torch.tensor(F['cat_h'][o], device=dev),
                          torch.tensor(F['cat_l'][o], device=dev), torch.tensor(y[o], device=dev))
    Xt, Ht, Lt, Yt = pk(Ftr, ytr, otr); Xv, Hv, Lv, Yv = pk(Fva, yva, ova)
    m = BlockMatcher(Xt.shape[-1], vocab, d=d, hid=hid, pdrop=pdrop).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((Btr + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, nb):
            ri = (perm[k:k + nb][:, None] * 8 + torch.arange(8)).reshape(-1)
            M, _ = m(Xt[ri], Ht[ri], Lt[ri], sinkhorn_iter=sink_train)
            tgt = Yt[ri].reshape(-1, 8)
            loss = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1)) + colw * Fn.cross_entropy(
                M.transpose(1, 2).reshape(-1, 8), torch.argsort(tgt, 1).reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Mr, Ms = m(Xv, Hv, Lv, sinkhorn_iter=30)
        tra = None
        if ret_train:
            sub = torch.arange(min(Btr, 400) * 8)
            _, Mts = m(Xt[sub], Ht[sub], Lt[sub], 30)
            tra = adjusted(credit_from_scores(Mts.reshape(-1, 8).cpu().numpy(), Yt[sub].cpu().numpy()))
    inv = np.argsort(ova)
    return Mr.reshape(-1, 8).cpu().numpy()[inv], Ms.reshape(-1, 8).cpu().numpy()[inv], tra


def run(tag, loo=True, seeds=(0,), nf=5, cache_feats=True, **kw):
    t0 = time.time()
    D = load('train.csv'); df = D['df']; y = D['y']
    fold = make_folds(df, nf)
    oof = np.zeros((len(df), 8), np.float32); oofraw = np.zeros((len(df), 8), np.float32); pf = []
    for f in range(nf):
        tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
        cp = f'{CACHE}/f3_fold{f}_loo{int(loo)}.npz'
        if cache_feats and os.path.exists(cp):
            z = np.load(cp, allow_pickle=True)
            Ftr = {k: z['tr_' + k] for k in ('X', 'cat_h', 'cat_l', 'block')}
            Fva = {k: z['va_' + k] for k in ('X', 'cat_h', 'cat_l', 'block')}
            voc = z['voc'].item()
        else:
            fb = FB3().fit(D, tri)
            Ftr = fb.transform(D, tri, loo=loo); Fva = fb.transform(D, vai)
            voc = {k: len(v) for k, v in fb.vocab.items()}
            if cache_feats:
                np.savez_compressed(cp, voc=np.array(voc, dtype=object),
                                    **{'tr_' + k: v for k, v in Ftr.items()},
                                    **{'va_' + k: v for k, v in Fva.items()})
        acc = np.zeros((len(vai), 8)); accr = np.zeros((len(vai), 8)); tra = None
        for s in seeds:
            rw, sk, t_ = fit_fold(Ftr, y[tri], Fva, y[vai], voc, seed=s, ret_train=(s == seeds[0]), **kw)
            acc += sk; accr += rw
            if t_ is not None:
                tra = t_
        oof[vai] = acc / len(seeds); oofraw[vai] = accr / len(seeds)
        a = adjusted(credit_from_scores(oof[vai], y[vai])); pf.append(a)
        print(f'  [{tag}] fold {f} adj={a:.4f} train={tra:.4f} [{time.time()-t0:.0f}s]', flush=True)
    c = credit_from_scores(oof, y); hum = df['species'].values == 'human'
    print(f'== {tag}: OOF={adjusted(c):.4f} human={adjusted(c[hum]):.4f} minfold={min(pf):.4f} '
          f'proxy={0.75*adjusted(c)+0.25*min(pf):.4f} nfeat={Ftr["X"].shape[-1]}', flush=True)
    return oof, oofraw, adjusted(c)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v3'); ap.add_argument('--loo', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=14); ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--pdrop', type=float, default=0.15); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hid', type=int, default=192)
    a = ap.parse_args()
    o, orw, s = run(a.tag, loo=bool(a.loo), seeds=tuple(range(a.seeds)), epochs=a.epochs,
                    pdrop=a.pdrop, wd=a.wd, hid=a.hid)
    np.save(f'{CACHE}/oof_{a.tag}.npy', o)
    np.save(f'{CACHE}/oofraw_{a.tag}.npy', orw)

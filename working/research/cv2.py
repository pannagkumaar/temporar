import sys, os, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from feats2 import FB2
from model2 import BlockMatcher2
from model import block_order

torch.set_num_threads(8)


def make_folds(df, nf=5, seed=7):
    samples = df['sample_id'].values
    uq = np.array(sorted(set(samples)))
    perm = np.random.default_rng(seed).permutation(len(uq))
    fo = {s: perm[i] % nf for i, s in enumerate(uq)}
    return np.array([fo[s] for s in samples])


def fit_fold(Ftr, ytr, Fva, yva, vocab, seed=0, epochs=14, lr=3e-3, wd=1e-4,
             sink_train=6, dev='cpu', hid=224, d=48, dr=24, pdrop=0.15, use_res=True,
             colw=0.5, nb=96, ret_train=False):
    torch.manual_seed(seed); np.random.seed(seed)
    otr, Btr = block_order(Ftr['block']); ova, Bva = block_order(Fva['block'])

    def pack(F, y, o):
        t = dict(X=torch.tensor(F['X'][o], device=dev),
                 ch=torch.tensor(F['cat_h'][o], device=dev),
                 cl=torch.tensor(F['cat_l'][o], device=dev),
                 y=torch.tensor(y[o], device=dev))
        if use_res:
            t['rh'] = torch.tensor(F['res_h'][o], device=dev)
            t['rl'] = torch.tensor(F['res_l'][o], device=dev)
        else:
            t['rh'] = t['rl'] = None
        return t

    T = pack(Ftr, ytr, otr); V = pack(Fva, yva, ova)
    m = BlockMatcher2(T['X'].shape[-1], vocab, d=d, dr=dr, hid=hid, pdrop=pdrop,
                      use_res=use_res).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((Btr + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)
    sl = lambda t, i: (None if t is None else t[i])
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, nb):
            bi = perm[k:k + nb]
            ri = (bi[:, None] * 8 + torch.arange(8)).reshape(-1)
            M, _ = m(T['X'][ri], T['ch'][ri], T['cl'][ri], sl(T['rh'], ri), sl(T['rl'], ri),
                     sinkhorn_iter=sink_train)
            tgt = T['y'][ri].reshape(-1, 8)
            loss = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1)) + \
                colw * Fn.cross_entropy(M.transpose(1, 2).reshape(-1, 8),
                                        torch.argsort(tgt, dim=1).reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Mr, Ms = m(V['X'], V['ch'], V['cl'], V['rh'], V['rl'], sinkhorn_iter=30)
        tr_adj = None
        if ret_train:
            sub = torch.arange(min(Btr, 400) * 8)
            Mt, Mts = m(T['X'][sub], T['ch'][sub], T['cl'][sub], sl(T['rh'], sub), sl(T['rl'], sub), 30)
            tr_adj = adjusted(credit_from_scores(Mts.reshape(-1, 8).cpu().numpy(),
                                                 T['y'][sub].cpu().numpy()))
    inv = np.argsort(ova)
    return (Mr.reshape(-1, 8).cpu().numpy()[inv], Ms.reshape(-1, 8).cpu().numpy()[inv], m, tr_adj)


def run(tag, use_res=True, epochs=14, seeds=(0,), nf=5, loo=True, **kw):
    t0 = time.time()
    D = load('train.csv'); df = D['df']; y = D['y']
    fold = make_folds(df, nf)
    oof = np.zeros((len(df), 8), np.float32); oofr = np.zeros((len(df), 8), np.float32)
    per_fold = []
    for f in range(nf):
        tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
        fb = FB2(112, use_res=use_res).fit(D, tri)
        Ftr = fb.transform(D, tri, loo=loo); Fva = fb.transform(D, vai)
        voc = {k: len(v) for k, v in fb.vocab.items()}
        acc = np.zeros((len(vai), 8)); accr = np.zeros((len(vai), 8)); tra = None
        for s in seeds:
            raw, sk, _, tra = fit_fold(Ftr, y[tri], Fva, y[vai], voc, seed=s, epochs=epochs,
                                       use_res=use_res, ret_train=(s == seeds[0]), **kw)
            acc += sk; accr += raw
        oof[vai] = acc / len(seeds); oofr[vai] = accr / len(seeds)
        a = adjusted(credit_from_scores(oof[vai], y[vai]))
        per_fold.append(a)
        print(f'  [{tag}] fold {f} adj={a:.4f} raw={adjusted(credit_from_scores(oofr[vai],y[vai])):.4f}'
              f' train={tra:.4f} [{time.time()-t0:.0f}s]', flush=True)
    c = credit_from_scores(oof, y); hum = df['species'].values == 'human'
    print(f'== {tag}: OOF={adjusted(c):.4f} human={adjusted(c[hum]):.4f} '
          f'minfold={min(per_fold):.4f} proxy={0.75*adjusted(c)+0.25*min(per_fold):.4f}', flush=True)
    return oof, c


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='both')
    ap.add_argument('--epochs', type=int, default=14)
    a = ap.parse_args()
    if a.mode in ('nores', 'both'):
        o, _ = run('v2-nores', use_res=False, epochs=a.epochs)
        np.save('working/cache/oof_v2_nores.npy', o)
    if a.mode in ('res', 'both'):
        o, _ = run('v2-res', use_res=True, epochs=a.epochs)
        np.save('working/cache/oof_v2_res.npy', o)

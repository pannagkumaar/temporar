import sys, os, time, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from feats import FeatureBuilder
from model import BlockMatcher, log_sinkhorn, block_order

torch.set_num_threads(8)


def fit_fold(Ftr, ytr, Fva, yva, vocab, seed=0, epochs=14, lr=3e-3, wd=1e-4,
             sink_train=6, dev='cpu', verbose=False, hid=192, d=48, pdrop=0.15):
    torch.manual_seed(seed); np.random.seed(seed)
    otr, Btr = block_order(Ftr['block'])
    ova, Bva = block_order(Fva['block'])

    def pack(F, y, order, dev):
        X = torch.tensor(F['X'][order], device=dev)
        ch = torch.tensor(F['cat_h'][order], device=dev)
        cl = torch.tensor(F['cat_l'][order], device=dev)
        yy = torch.tensor(y[order], device=dev)
        return X, ch, cl, yy

    Xtr, Htr, Ltr, Ytr = pack(Ftr, ytr, otr, dev)
    Xva, Hva, Lva, Yva = pack(Fva, yva, ova, dev)
    m = BlockMatcher(Xtr.shape[-1], vocab, d=d, hid=hid, pdrop=pdrop).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    nb = 96
    steps = epochs * ((Btr + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, nb):
            bi = perm[k:k + nb]
            ridx = (bi[:, None] * 8 + torch.arange(8)).reshape(-1)
            M, _ = m(Xtr[ridx], Htr[ridx], Ltr[ridx], sinkhorn_iter=sink_train)
            tgt = Ytr[ridx].reshape(-1, 8)
            lr_ = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1))
            # column loss: each candidate should pick its heavy (bijection, both directions)
            colt = torch.argsort(tgt, dim=1)
            lc_ = Fn.cross_entropy(M.transpose(1, 2).reshape(-1, 8), colt.reshape(-1))
            loss = lr_ + 0.5 * lc_
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Mraw, Msk = m(Xva, Hva, Lva, sinkhorn_iter=30)
    inv = np.argsort(ova)
    raw = Mraw.reshape(-1, 8).cpu().numpy()[inv]
    sk = Msk.reshape(-1, 8).cpu().numpy()[inv]
    return raw, sk, m


def main():
    t0 = time.time()
    D = load('train.csv'); df = D['df']; y = D['y']
    samples = df['sample_id'].values
    uq = np.array(sorted(set(samples)))
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(uq))
    NF = 5
    fold_of = {s: perm[i] % NF for i, s in enumerate(uq)}
    fold = np.array([fold_of[s] for s in samples])

    oof_raw = np.zeros((len(df), 8), dtype=np.float32)
    oof_sk = np.zeros((len(df), 8), dtype=np.float32)
    for f in range(NF):
        tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
        fb = FeatureBuilder(110).fit(D, tri)
        Ftr = fb.transform(D, tri); Fva = fb.transform(D, vai)
        vocab = {k: len(v) for k, v in fb.vocab.items()}
        raw, sk, _ = fit_fold(Ftr, y[tri], Fva, y[vai], vocab, seed=f)
        oof_raw[vai] = raw; oof_sk[vai] = sk
        cr = credit_from_scores(raw, y[vai]); cs = credit_from_scores(sk, y[vai])
        print(f'fold {f}  n={len(vai)}  raw={adjusted(cr):.4f}  sinkhorn={adjusted(cs):.4f}  '
              f'top1={(cs==1).mean():.3f}  [{time.time()-t0:.0f}s]', flush=True)
    cr = credit_from_scores(oof_raw, y); cs = credit_from_scores(oof_sk, y)
    print(f'OOF raw={adjusted(cr):.4f}  sinkhorn={adjusted(cs):.4f}')
    hum = df['species'].values == 'human'
    print(f'human-only  raw={adjusted(cr[hum]):.4f}  sinkhorn={adjusted(cs[hum]):.4f}')
    # per-sample spread (worst-family proxy)
    per = pd.DataFrame({'s': samples, 'c': cs}).groupby('s')['c'].agg(['mean', 'size'])
    per['adj'] = (2 * per['mean'] - 1).clip(0, 1)
    big = per[per['size'] >= 100]
    print('per-sample adjusted: min %.3f q10 %.3f med %.3f max %.3f' %
          (big['adj'].min(), big['adj'].quantile(.10), big['adj'].median(), big['adj'].max()))
    np.save('working/cache/oof_raw.npy', oof_raw); np.save('working/cache/oof_sk.npy', oof_sk)
    np.save('working/cache/fold.npy', fold)


if __name__ == '__main__':
    main()

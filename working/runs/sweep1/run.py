"""GPU job 5 -- spend the unused delivery budget, and close two untested free variables.

The delivered plan runs in 14.4 min of a 90-minute budget. That headroom is real score left on
the table, but only if more of it actually buys something. Arms, all on the incumbent's folds
and embeddings (mean + CDR3-span pooling, AntiBERTa2 + IgBert + ESM-2):

  seed curve 1/3/8/16      - how far does averaging keep paying?
  human_only               - the test set is 100% human and the mouse/rat rows score 0.06;
                             are they helping or adding gradient noise? Free at delivery.
  marginal-averaging       - average per-seed marginals instead of per-seed logits
  head capacity/regularisation - de, edrop, hid, epochs, colw
  blend                    - average logits across the diverse configs, not just seeds

Incumbent folds [0.6026, 0.5863, 0.5562, 0.5889, 0.5345], OOF 0.5739, human 0.6067.
"""
import os, sys, time, json
import numpy as np

ART = os.environ.get('ARTIFACT_ROOT', './artifacts')
DATA = os.environ.get('DATA_ROOT', './dataset_public')
PROJ = os.environ.get('PROJECT_ROOT', '.')
os.makedirs(ART, exist_ok=True)
sys.path.insert(0, os.path.join(PROJ, 'working/research'))
os.chdir(PROJ)
log = lambda *a: print(*a, flush=True)

import torch
if not torch.cuda.is_available():
    log('FATAL no cuda'); sys.exit(3)
a = torch.randn(2048, 2048, device='cuda'); (a @ a).sum().item()
b = torch.randn(8, 512, 512, device='cuda', dtype=torch.bfloat16)
torch.bmm(b, b).float().sum().item(); torch.cuda.synchronize()
log('CUDA OK', torch.cuda.get_device_name(0))

import torch.nn as nn, torch.nn.functional as Fn
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from common import load, credit_from_scores, adjusted
from feats3 import FB3
from model4 import BlockMatcher4
from model import block_order
from permmarg import marginals
from cv3 import make_folds

DEV = 'cuda'; t00 = time.time(); RES = {}
MAXLEN = 152
SPECS = [('antiberta2', 'alchemab/antiberta2', BertTokenizer),
         ('igbert', 'Exscientia/IgBert', AutoTokenizer),
         ('esm2', 'facebook/esm2_t33_650M_UR50D', AutoTokenizer)]
INCUMBENT = [0.6026, 0.5863, 0.5562, 0.5889, 0.5345]

D = load(os.path.join(DATA, 'train.csv'))
df = D['df']; y = D['y']
fold = make_folds(df, 5)
hum = df['species'].values == 'human'
heavy = sorted(set(df['heavy_chain_aa'].astype(str)))
light = sorted(set(D['aas'].reshape(-1)))
hi = {s: i for i, s in enumerate(heavy)}; li = {s: i for i, s in enumerate(light)}
IDX_H = np.array([hi[s] for s in df['heavy_chain_aa'].astype(str)], np.int64)
IDX_L = np.array([[li[s] for s in row] for row in D['aas']], np.int64)
hcd = dict(zip(df['heavy_chain_aa'].astype(str), df['heavy_cdr3_aa']))
lcd = {}
for i in range(len(df)):
    for j in range(8):
        lcd[D['aas'][i, j]] = D['lc'][i, j]
H_SPAN = np.array([[max(s.rfind(hcd[s]), 0), max(s.rfind(hcd[s]), 0) + len(hcd[s])] for s in heavy])
L_SPAN = np.array([[max(s.rfind(lcd[s]), 0), max(s.rfind(lcd[s]), 0) + len(lcd[s])] for s in light])


def tokenize(tok, seqs):
    e = tok([' '.join(list(s)) for s in seqs], return_tensors='pt', padding='max_length',
            truncation=True, max_length=MAXLEN)
    return e['input_ids'], e['attention_mask']


@torch.no_grad()
def pooled(model, ids, mask, spans, bs=96):
    hd = model.config.hidden_size
    out = np.zeros((len(ids), hd * 2), np.float32)
    ar = torch.arange(MAXLEN, device=DEV)[None, :]
    for i in range(0, len(ids), bs):
        ii = ids[i:i + bs].to(DEV); mm = mask[i:i + bs].to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(input_ids=ii, attention_mask=mm).last_hidden_state.float()
        m = mm.unsqueeze(-1).float()
        sp = torch.tensor(spans[i:i + len(ii)], device=DEV)
        cm = ((ar >= sp[:, 0:1] + 1) & (ar < sp[:, 1:2] + 1)).unsqueeze(-1).float()
        out[i:i + len(ii)] = torch.cat([(h * m).sum(1) / m.sum(1),
                                        (h * cm).sum(1) / cm.sum(1).clamp(min=1)], 1).cpu().numpy()
    return out


EMB = {}
for nm, repo, tc in SPECS:
    t0 = time.time()
    tok = tc.from_pretrained(repo)
    mdl = AutoModel.from_pretrained(repo).cuda().eval()
    hid, hm = tokenize(tok, heavy); lid, lm = tokenize(tok, light)
    EMB[nm] = (pooled(mdl, hid, hm, H_SPAN), pooled(mdl, lid, lm, L_SPAN))
    log(f'  embed {nm} {EMB[nm][0].shape} [{time.time()-t0:.0f}s]')
    del mdl; torch.cuda.empty_cache()

FEAT = {}
for f in range(5):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    fb = FB3().fit(D, tri)
    FEAT[f] = (fb.transform(D, tri, loo=True), fb.transform(D, vai),
               {k: len(v) for k, v in fb.vocab.items()}, tri, vai)
log(f'features done [{time.time()-t00:.0f}s]')

# human-only variant needs its own tables (fitted on human training rows only)
FEAT_H = {}
for f in range(5):
    tri = np.where((fold != f) & hum)[0]; vai = np.where(fold == f)[0]
    fb = FB3().fit(D, tri)
    FEAT_H[f] = (fb.transform(D, tri, loo=True), fb.transform(D, vai),
                 {k: len(v) for k, v in fb.vocab.items()}, tri, vai)
log(f'human-only features done [{time.time()-t00:.0f}s]')


def zs(A, rows):
    A = A.astype(np.float32); s = A[rows]
    return (A - s.mean(0, keepdims=True)) / (s.std(0, keepdims=True) + 1e-5)


def fit(Ftr, ytr, Fva, yva, voc, EH, EL, ihtr, iltr, ihva, ilva, seed, epochs=14, lr=3e-3,
        wd=1e-4, hid=192, d=48, pdrop=0.15, de=64, edrop=0.3, colw=0.5, nb=96):
    torch.manual_seed(seed); np.random.seed(seed)
    otr, Btr = block_order(Ftr['block']); ova, _ = block_order(Fva['block'])
    T = lambda a, o: torch.tensor(a[o], device=DEV)
    Xt, Ht, Lt, Yt = T(Ftr['X'], otr), T(Ftr['cat_h'], otr), T(Ftr['cat_l'], otr), T(ytr, otr)
    Xv, Hv, Lv = T(Fva['X'], ova), T(Fva['cat_h'], ova), T(Fva['cat_l'], ova)
    IHt, ILt, IHv, ILv = T(ihtr, otr), T(iltr, otr), T(ihva, ova), T(ilva, ova)
    m = BlockMatcher4(Xt.shape[-1], voc, EH, EL, d=d, hid=hid, pdrop=pdrop, de=de,
                      edrop=edrop).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((Btr + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, nb):
            ri = (perm[k:k + nb][:, None] * 8 + torch.arange(8)).reshape(-1).to(DEV)
            M, _ = m(Xt[ri], Ht[ri], Lt[ri], IHt[ri], ILt[ri], sinkhorn_iter=6)
            tg = Yt[ri].reshape(-1, 8)
            loss = Fn.cross_entropy(M.reshape(-1, 8), tg.reshape(-1))
            if colw > 0:
                loss = loss + colw * Fn.cross_entropy(
                    M.transpose(1, 2).reshape(-1, 8), torch.argsort(tg, 1).reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Mr, _ = m(Xv, Hv, Lv, IHv, ILv, sinkhorn_iter=30)
    return Mr.reshape(-1, 8).cpu().numpy()[np.argsort(ova)]


def score_all(raw, T=1.25):
    o, B = block_order(df['block_id'].values); inv = np.argsort(o)
    p = marginals(raw[o].reshape(B, 8, 8), T)
    sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
    return (adjusted(credit_from_scores(sc, y)),
            adjusted(credit_from_scores(sc[hum], y[hum])))


def fold_sc(rawv, blocks, yv, T=1.25):
    o, B = block_order(blocks); inv = np.argsort(o)
    p = marginals(rawv[o].reshape(B, 8, 8), T)
    return adjusted(credit_from_scores(np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv], yv))


def run(tag, nseeds=3, human_only=False, avg_marginal=False, **kw):
    t0 = time.time()
    raw = np.zeros((len(df), 8), np.float32); pf = []
    src = FEAT_H if human_only else FEAT
    for f in range(5):
        Ftr, Fva, voc, tri, vai = src[f]
        rows_h = np.unique(IDX_H[tri]); rows_l = np.unique(IDX_L[tri].reshape(-1))
        EH = np.concatenate([zs(EMB[n][0], rows_h) for n, _, _ in SPECS], 1)
        EL = np.concatenate([zs(EMB[n][1], rows_l) for n, _, _ in SPECS], 1)
        if avg_marginal:
            o, B = block_order(Fva['block']); inv = np.argsort(o)
            acc = np.zeros((len(vai), 8))
            for s in range(nseeds):
                r1 = fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL, IDX_H[tri], IDX_L[tri],
                         IDX_H[vai], IDX_L[vai], seed=s, **kw)
                acc += marginals(r1[o].reshape(B, 8, 8), 1.25).reshape(-1, 8)[inv]
            raw[vai] = np.log(np.maximum(acc / nseeds, 1e-300))
            pf.append(adjusted(credit_from_scores(raw[vai], y[vai])))
        else:
            acc = np.zeros((len(vai), 8))
            for s in range(nseeds):
                acc += fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL, IDX_H[tri], IDX_L[tri],
                           IDX_H[vai], IDX_L[vai], seed=s, **kw)
            raw[vai] = acc / nseeds
            pf.append(fold_sc(raw[vai], Fva['block'], y[vai]))
    if avg_marginal:
        ov = adjusted(credit_from_scores(raw, y)); hu = adjusted(credit_from_scores(raw[hum], y[hum]))
    else:
        ov, hu = score_all(raw)
    RES[tag] = dict(oof=ov, human=hu, folds=pf, delta_vs_incumbent=round(ov - 0.5739, 4))
    log(f'ARM {tag:22s} OOF={ov:.4f} human={hu:.4f} d={ov-0.5739:+.4f} '
        f'folds={[round(x,3) for x in pf]} [{time.time()-t0:.0f}s]')
    json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
    np.save(f'{ART}/rawoof_{tag}.npy', raw)
    return raw


BASE = run('ctrl_seeds3', nseeds=3)
run('seeds1', nseeds=1)
S8 = run('seeds8', nseeds=8)
run('seeds16', nseeds=16)
run('human_only', nseeds=3, human_only=True)
run('avg_marginal8', nseeds=8, avg_marginal=True)
V = {}
for tag, kw in [('de128', dict(de=128)), ('edrop45', dict(edrop=0.45)), ('hid320', dict(hid=320)),
                ('ep22', dict(epochs=22)), ('colw1', dict(colw=1.0)), ('colw0', dict(colw=0.0)),
                ('d64emb', dict(d=64)), ('pdrop25', dict(pdrop=0.25))]:
    V[tag] = run(tag, nseeds=3, **kw)

# diverse blend: average logits across configurations rather than seeds of one configuration
names = sorted(V.keys())
bl = (S8 * 2.0 + sum(V[n] for n in names)) / (2.0 + len(names))
ov, hu = score_all(bl)
RES['blend_diverse'] = dict(oof=ov, human=hu, delta_vs_incumbent=round(ov - 0.5739, 4),
                            members=['seeds8'] + names)
log(f'ARM {"blend_diverse":22s} OOF={ov:.4f} human={hu:.4f} d={ov-0.5739:+.4f}')
np.save(f'{ART}/rawoof_blend_diverse.npy', bl)
json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
log('DONE', json.dumps(RES)[:1500])

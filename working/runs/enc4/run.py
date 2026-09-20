"""GPU job 8 -- does a FOURTH encoder still pay?

Each encoder added so far paid: IgBert +0.057, +AntiBERTa2 +0.011, +ESM-2 +0.010 OOF.
Two candidates for a fourth, both cheap and tokenizer-safe:
  antiberta2-cssp  - same architecture as AntiBERTa2 but contrastively pretrained against
                     STRUCTURE, so it may carry VH/VL interface geometry the sequence-only
                     models cannot see. This is the one with a mechanism behind it.
  IgBert_unpaired  - same architecture as IgBert, different pretraining corpus; expected to be
                     highly correlated with IgBert and therefore a near-null control.

Original header follows.
GPU job 7 -- confirm the winning combination and fix the delivery recipe.

sweep1 found two independent wins over the 3-seed control (OOF 0.5768 / human 0.6079):
  seeds 3 -> 8 -> 16        +0.0058 / +0.0084 OOF
  embedding dropout 0.3->0.45  +0.0073 OOF, at only 3 seeds
and rejected de128, hid320, ep22, colw1, marginal-averaging and human-only training.

Regularisation and ensembling are different mechanisms so they should compose, but that is an
assumption, not a measurement. This job measures the combination directly and picks the exact
constants the delivered solution.py will carry. Every arm is 5-fold, same folds, same
embeddings (mean + CDR3-span pooling, AntiBERTa2 + IgBert + ESM-2), exact-marginal decode.
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
         ('esm2', 'facebook/esm2_t33_650M_UR50D', AutoTokenizer),
         ('cssp', 'alchemab/antiberta2-cssp', BertTokenizer),
         ('igbertu', 'Exscientia/IgBert_unpaired', AutoTokenizer)]

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
    try:
        tok = tc.from_pretrained(repo)
        mdl = AutoModel.from_pretrained(repo).cuda().eval()
    except Exception as e:
        log(f'  SKIP {nm} ({repo}): {type(e).__name__}: {str(e)[:200]}')
        RES[f'skip_{nm}'] = str(e)[:300]
        continue
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
            loss = Fn.cross_entropy(M.reshape(-1, 8), tg.reshape(-1)) + colw * Fn.cross_entropy(
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
    return (adjusted(credit_from_scores(sc, y)), adjusted(credit_from_scores(sc[hum], y[hum])))


def fold_sc(rawv, blocks, yv, T=1.25):
    o, B = block_order(blocks); inv = np.argsort(o)
    p = marginals(rawv[o].reshape(B, 8, 8), T)
    return adjusted(credit_from_scores(np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv], yv))


def build(names, rows_h, rows_l):
    EH = np.concatenate([zs(EMB[n][0], rows_h) for n in names], 1)
    EL = np.concatenate([zs(EMB[n][1], rows_l) for n in names], 1)
    return EH, EL


def run(tag, nseeds, names=('antiberta2', 'igbert', 'esm2'), **kw):
    t0 = time.time()
    raw = np.zeros((len(df), 8), np.float32); pf = []
    for f in range(5):
        Ftr, Fva, voc, tri, vai = FEAT[f]
        rows_h = np.unique(IDX_H[tri]); rows_l = np.unique(IDX_L[tri].reshape(-1))
        EH, EL = build(names, rows_h, rows_l)
        acc = np.zeros((len(vai), 8))
        for s in range(nseeds):
            acc += fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL, IDX_H[tri], IDX_L[tri],
                       IDX_H[vai], IDX_L[vai], seed=s, **kw)
        raw[vai] = acc / nseeds
        pf.append(fold_sc(raw[vai], Fva['block'], y[vai]))
    ov, hu = score_all(raw)
    RES[tag] = dict(oof=ov, human=hu, folds=pf, nseeds=nseeds, **{k: v for k, v in kw.items()})
    log(f'ARM {tag:22s} OOF={ov:.4f} human={hu:.4f} folds={[round(x,3) for x in pf]} '
        f'[{time.time()-t0:.0f}s]')
    json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
    np.save(f'{ART}/rawoof_{tag}.npy', raw)
    return raw



EDROP = 0.45
B3 = ('antiberta2', 'igbert', 'esm2')
run('ctrl_3enc', 8, names=B3, edrop=EDROP)
if 'cssp' in EMB:
    run('plus_cssp', 8, names=B3 + ('cssp',), edrop=EDROP)
if 'igbertu' in EMB:
    run('plus_igbertu', 8, names=B3 + ('igbertu',), edrop=EDROP)
if 'cssp' in EMB and 'igbertu' in EMB:
    run('all_5enc', 8, names=B3 + ('cssp', 'igbertu'), edrop=EDROP)
json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
log('DONE', json.dumps(RES)[:1500])

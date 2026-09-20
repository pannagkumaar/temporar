"""GPU job 6 -- the structural challenger: within-sample re-blocking augmentation.

Different hypothesis, not a new name for an old feature. The incumbent assumes the 3797 shipped
blocks are the training set. They are one arbitrary partition of each sample's 320 cells out of
~2.5e15 valid ones, so the incumbent sees every negative pair in fixed company and every
block-relative feature (SHM rank, z-score, clonal profile) at exactly one realisation per cell.
Re-partitioning within sample yields new blocks with identical structure - 8 real cells, one
donor, a real bijection, same-donor decoys - and multiplies both the negative pairs and the
block contexts the head is trained on. No fabricated data, no external data.

Verified locally: bijection holds in every re-blocked block, targets balanced 3684 per slot,
true partner preserved, candidate slots alphabetical by light_chain_code as the challenge
builds them, 3% of rows dropped where a block would have contained two identical light chains.

Arms: K=0 (control, same seeds), K=2, K=6 extra partitions per fold.
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
from reblock import reblock

DEV = 'cuda'; t00 = time.time(); RES = {}
MAXLEN = 152
KMAX = 6
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
    log(f'  embed {nm} [{time.time()-t0:.0f}s]')
    del mdl; torch.cuda.empty_cache()


def zs(A, rows):
    A = A.astype(np.float32); s = A[rows]
    return (A - s.mean(0, keepdims=True)) / (s.std(0, keepdims=True) + 1e-5)


# ---- per-fold: base features + KMAX re-blocked variants ----
PACK = {}
for f in range(5):
    t0 = time.time()
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    fb = FB3().fit(D, tri)                       # tables fitted on this fold's training rows only
    base = fb.transform(D, tri, loo=True)
    Fva = fb.transform(D, vai)
    voc = {k: len(v) for k, v in fb.vocab.items()}
    variants = []
    for k in range(KMAX):
        R = reblock(D, tri, seed=1000 * f + k)
        Fk = fb.transform(R, np.arange(len(R['df'])), loo=True)
        variants.append((Fk, R['y'], IDX_H[R['abs_rows']], IDX_L[R['abs_rows']]))
    PACK[f] = (base, Fva, voc, tri, vai, variants)
    log(f'  fold {f} features + {KMAX} re-blockings [{time.time()-t0:.0f}s]')


def fit_multi(parts, Fva, yva, voc, EH, EL, ihva, ilva, seed, epochs=14, lr=3e-3, wd=1e-4,
              hid=192, d=48, pdrop=0.15, de=64, edrop=0.3, colw=0.5, nb=96):
    """parts: list of (F, y, ih, il). Blocks from every part are pooled into one epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    T = lambda a: torch.tensor(a, device=DEV)
    packs = []
    for F, yy, ih, il in parts:
        o, B = block_order(F['block'])
        packs.append((T(F['X'][o]), T(F['cat_h'][o]), T(F['cat_l'][o]), T(yy[o]),
                      T(ih[o]), T(il[o]), B))
    offs = np.cumsum([0] + [p[6] for p in packs])
    TOT = int(offs[-1])
    ova, _ = block_order(Fva['block'])
    Xv, Hv, Lv = T(Fva['X'][ova]), T(Fva['cat_h'][ova]), T(Fva['cat_l'][ova])
    IHv, ILv = T(ihva[ova]), T(ilva[ova])
    m = BlockMatcher4(packs[0][0].shape[-1], voc, EH, EL, d=d, hid=hid, pdrop=pdrop, de=de,
                      edrop=edrop).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((TOT + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(TOT, generator=g)
        for k in range(0, TOT, nb):
            gb = perm[k:k + nb].numpy()
            pi = np.searchsorted(offs, gb, 'right') - 1
            loss = 0.0
            for p in np.unique(pi):
                bi = torch.tensor(gb[pi == p] - offs[p])
                Xt, Ht, Lt, Yt, IHt, ILt, _ = packs[p]
                ri = (bi[:, None] * 8 + torch.arange(8)).reshape(-1).to(DEV)
                M, _ = m(Xt[ri], Ht[ri], Lt[ri], IHt[ri], ILt[ri], sinkhorn_iter=6)
                tg = Yt[ri].reshape(-1, 8)
                w = len(bi) / len(gb)
                loss = loss + w * (Fn.cross_entropy(M.reshape(-1, 8), tg.reshape(-1)) +
                                   colw * Fn.cross_entropy(M.transpose(1, 2).reshape(-1, 8),
                                                           torch.argsort(tg, 1).reshape(-1)))
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


def run(tag, K, seeds=(0, 1), **kw):
    t0 = time.time()
    raw = np.zeros((len(df), 8), np.float32); pf = []
    for f in range(5):
        base, Fva, voc, tri, vai, variants = PACK[f]
        rows_h = np.unique(IDX_H[tri]); rows_l = np.unique(IDX_L[tri].reshape(-1))
        EH = np.concatenate([zs(EMB[n][0], rows_h) for n, _, _ in SPECS], 1)
        EL = np.concatenate([zs(EMB[n][1], rows_l) for n, _, _ in SPECS], 1)
        parts = [(base, y[tri], IDX_H[tri], IDX_L[tri])] + variants[:K]
        acc = np.zeros((len(vai), 8))
        for s in seeds:
            acc += fit_multi(parts, Fva, y[vai], voc, EH, EL, IDX_H[vai], IDX_L[vai],
                             seed=s, **kw)
        raw[vai] = acc / len(seeds)
        sc = fold_sc(raw[vai], Fva['block'], y[vai]); pf.append(sc)
        log(f'  [{tag}] fold {f} adj={sc:.4f} (incumbent {INCUMBENT[f]:.4f}, '
            f'{sc-INCUMBENT[f]:+.4f}) [{time.time()-t0:.0f}s]')
    ov, hu = score_all(raw)
    RES[tag] = dict(oof=ov, human=hu, folds=pf, delta=round(ov - 0.5739, 4))
    log(f'ARM {tag:14s} OOF={ov:.4f} human={hu:.4f} d={ov-0.5739:+.4f} [{time.time()-t0:.0f}s]')
    json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
    np.save(f'{ART}/rawoof_{tag}.npy', raw)
    return raw


run('K0_ctrl', 0)
run('K2', 2)
run('K6', 6)
run('K6_ep20', 6, epochs=20)
json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
log('DONE', json.dumps(RES)[:1200])

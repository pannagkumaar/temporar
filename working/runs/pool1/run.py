"""GPU job 3.

1. DELIVERY BLOCKER: verify BertTokenizer reproduces RoFormerTokenizer's input_ids for
   AntiBERTa2. Delivery uses BertTokenizer to avoid an rjieba dependency at evaluation time;
   if the ids differ, that substitution is invalid and must be reverted.
2. Region-pooled embeddings. Mean-pooling a 122-residue chain is dominated by framework.
   Pool separately over the CDR3 span, the 40 residues before it (FR3, carries the VH/VL
   interface positions) and the whole chain. Reference: mean-pooled 3-LM OOF = 0.5650.
3. Head hyperparameters on the winning pooling: seeds, projection width, embedding dropout.
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
MAXLEN = 160
SPECS = [('antiberta2', 'alchemab/antiberta2', BertTokenizer),
         ('igbert', 'Exscientia/IgBert', AutoTokenizer),
         ('esm2', 'facebook/esm2_t33_650M_UR50D', AutoTokenizer)]

# ---------- 1. tokenizer equivalence ----------
D = load(os.path.join(DATA, 'train.csv'))
df = D['df']; y = D['y']
probe = sorted(set(df['heavy_chain_aa'].astype(str)))[:400] + \
        sorted(set(D['aas'].reshape(-1)))[:400]
ref = AutoTokenizer.from_pretrained('alchemab/antiberta2')
alt = BertTokenizer.from_pretrained('alchemab/antiberta2')
txt = [' '.join(list(s)) for s in probe]
a1 = ref(txt, padding='max_length', truncation=True, max_length=MAXLEN)['input_ids']
a2 = alt(txt, padding='max_length', truncation=True, max_length=MAXLEN)['input_ids']
same = all(x == z for x, z in zip(a1, a2))
RES['antiberta2_berttokenizer_identical'] = bool(same)
log(f'TOKENIZER CHECK antiberta2 BertTokenizer == RoFormerTokenizer : {same} (n={len(probe)})')
if not same:
    d = [i for i, (x, z) in enumerate(zip(a1, a2)) if x != z][:3]
    for i in d:
        log(f'  mismatch {i}: {a1[i][:20]} vs {a2[i][:20]}')

# ---------- features ----------
fold = make_folds(df, 5)
FEAT = {}
for f in range(5):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    fb = FB3().fit(D, tri)
    FEAT[f] = (fb.transform(D, tri, loo=True), fb.transform(D, vai),
               {k: len(v) for k, v in fb.vocab.items()}, tri, vai)
log(f'features done [{time.time()-t00:.0f}s]')

heavy = sorted(set(df['heavy_chain_aa'].astype(str)))
light = sorted(set(D['aas'].reshape(-1)))
hi = {s: i for i, s in enumerate(heavy)}; li = {s: i for i, s in enumerate(light)}
IDX_H = np.array([hi[s] for s in df['heavy_chain_aa'].astype(str)], np.int64)
IDX_L = np.array([[li[s] for s in row] for row in D['aas']], np.int64)
# CDR3 spans (residue coords) for every unique sequence
hcd = dict(zip(df['heavy_chain_aa'].astype(str), df['heavy_cdr3_aa']))
lcd = {}
for i in range(len(df)):
    for j in range(8):
        lcd[D['aas'][i, j]] = D['lc'][i, j]
H_SPAN = np.array([[max(s.rfind(hcd[s]), 0), max(s.rfind(hcd[s]), 0) + len(hcd[s])] for s in heavy])
L_SPAN = np.array([[max(s.rfind(lcd[s]), 0), max(s.rfind(lcd[s]), 0) + len(lcd[s])] for s in light])


@torch.no_grad()
def encode(model, tok, seqs, spans, bs=96):
    """Returns (mean_all, mean_cdr3, mean_fr3) each (n, hidden)."""
    Hd = model.config.hidden_size
    A = np.zeros((len(seqs), Hd), np.float32)
    C = np.zeros((len(seqs), Hd), np.float32)
    Fr = np.zeros((len(seqs), Hd), np.float32)
    ar = torch.arange(MAXLEN, device=DEV)
    for i in range(0, len(seqs), bs):
        chunk = [' '.join(list(s)) for s in seqs[i:i + bs]]
        enc = tok(chunk, return_tensors='pt', padding='max_length', truncation=True,
                  max_length=MAXLEN)
        enc = {k: v.to(DEV) for k, v in enc.items()}
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(**enc).last_hidden_state.float()
        m = enc['attention_mask'].unsqueeze(-1).float()
        A[i:i + len(chunk)] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        sp = torch.tensor(spans[i:i + len(chunk)], device=DEV)
        # residue r sits at token r+1 (one [CLS])
        tokpos = ar[None, :]
        cm = ((tokpos >= sp[:, 0:1] + 1) & (tokpos < sp[:, 1:2] + 1)).unsqueeze(-1).float()
        fm = ((tokpos >= sp[:, 0:1] + 1 - 40) & (tokpos < sp[:, 0:1] + 1)).unsqueeze(-1).float()
        C[i:i + len(chunk)] = ((h * cm).sum(1) / cm.sum(1).clamp(min=1)).cpu().numpy()
        Fr[i:i + len(chunk)] = ((h * fm).sum(1) / fm.sum(1).clamp(min=1)).cpu().numpy()
    return A, C, Fr


EMB = {}
for nm, repo, tc in SPECS:
    t0 = time.time()
    tok = tc.from_pretrained(repo)
    mdl = AutoModel.from_pretrained(repo).cuda().eval()
    EMB[nm] = dict(h=encode(mdl, tok, heavy, H_SPAN), l=encode(mdl, tok, light, L_SPAN))
    log(f'  embed {nm} {time.time()-t0:.0f}s')
    del mdl; torch.cuda.empty_cache()


def zs(A):
    A = A.astype(np.float32)
    return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-5)


def build(parts):
    """parts: tuple of pooling indices (0=all,1=cdr3,2=fr3)"""
    EH = np.concatenate([zs(EMB[n]['h'][p]) for n in EMB for p in parts], 1)
    EL = np.concatenate([zs(EMB[n]['l'][p]) for n in EMB for p in parts], 1)
    return EH, EL


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
            tgt = Yt[ri].reshape(-1, 8)
            loss = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1)) + colw * Fn.cross_entropy(
                M.transpose(1, 2).reshape(-1, 8), torch.argsort(tgt, 1).reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Mr, _ = m(Xv, Hv, Lv, IHv, ILv, sinkhorn_iter=30)
    return Mr.reshape(-1, 8).cpu().numpy()[np.argsort(ova)]


def evaluate(tag, EH, EL, seeds=(0, 1, 2), **kw):
    raws = np.zeros((len(df), 8), np.float32); pf = []
    t0 = time.time()
    for f in range(5):
        Ftr, Fva, voc, tri, vai = FEAT[f]
        acc = np.zeros((len(vai), 8))
        for s in seeds:
            acc += fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL, IDX_H[tri], IDX_L[tri],
                       IDX_H[vai], IDX_L[vai], seed=s, **kw)
        raws[vai] = acc / len(seeds)
        o, B = block_order(Fva['block']); inv = np.argsort(o)
        p = marginals(raws[vai][o].reshape(B, 8, 8), 1.25)
        pf.append(adjusted(credit_from_scores(np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv],
                                              y[vai])))
    o, B = block_order(df['block_id'].values); inv = np.argsort(o)
    p = marginals(raws[o].reshape(B, 8, 8), 1.25)
    sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
    ov = adjusted(credit_from_scores(sc, y))
    hum = df['species'].values == 'human'
    hu = adjusted(credit_from_scores(sc[hum], y[hum]))
    RES[tag] = dict(oof=ov, human=hu, minfold=min(pf), folds=pf, proxy=0.75 * ov + 0.25 * min(pf))
    log(f'ARM {tag:34s} OOF={ov:.4f} human={hu:.4f} minfold={min(pf):.4f} '
        f'proxy={0.75*ov+0.25*min(pf):.4f} dim={EH.shape[1]} [{time.time()-t0:.0f}s]')
    json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
    np.save(f'{ART}/rawoof_{tag}.npy', raws)
    return ov


# ---------- 2. pooling ablation ----------
POOLS = [('mean', (0,)), ('mean+cdr3', (0, 1)), ('mean+cdr3+fr3', (0, 1, 2)), ('cdr3+fr3', (1, 2))]
best = ('mean', -1, None)
for tag, parts in POOLS:
    EH, EL = build(parts)
    v = evaluate(f'pool_{tag}', EH, EL)
    if v > best[1]:
        best = (tag, v, parts)
log(f'BEST POOLING {best[0]} {best[1]:.4f}')

# ---------- 3. head hyperparameters on the winner ----------
EH, EL = build(best[2])
for tag, kw in [('seeds6', dict(seeds=(0, 1, 2, 3, 4, 5))),
                ('de128', dict(de=128)),
                ('edrop45', dict(edrop=0.45)),
                ('hid320', dict(hid=320)),
                ('ep20', dict(epochs=20))]:
    evaluate(f'hp_{tag}', EH, EL, **kw)

json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
log('DONE', json.dumps(RES)[:2000])

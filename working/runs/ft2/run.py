"""GPU job 4 -- reopening fine-tuning, this time tested fairly.

The earlier probe (abpair-ft1 part B) was NOT a fair test of fine-tuning. It trained a
sequence-only cross-encoder with no access to the SHM clock or the gene-lift tables and
compared 0.489 against 0.570 for frozen-LM-PLUS-features. That comparison cannot separate
"fine-tuning does not help" from "sequences alone are weaker than sequences plus features".

The fair test, run here:
  fine-tune AntiBERTa2 on each fold's TRAINING blocks only -> extract the same
  (whole-chain, CDR3-span) pooled embeddings -> feed them into the SAME full model alongside
  frozen IgBert and frozen ESM-2 -> compare against the all-frozen incumbent on the same folds.

Incumbent (pool mean+cdr3, 3 seeds, exact marginals):
  OOF 0.5739  human 0.6067  folds [0.6026, 0.5863, 0.5562, 0.5889, 0.5345]

Fold 0 is run first and printed, so the run can be killed early if fine-tuning loses there.
Arms 'ctrl' (all frozen) reconfirms the incumbent inside this same job.
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
log('CUDA OK', torch.cuda.get_device_name(0),
    round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')

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
FT_REPO = 'alchemab/antiberta2'
FT_EPOCHS = int(os.environ.get('FT_EPOCHS', '3'))
FT_BLOCKS = 8
FROZEN = [('igbert', 'Exscientia/IgBert', AutoTokenizer),
          ('esm2', 'facebook/esm2_t33_650M_UR50D', AutoTokenizer)]

D = load(os.path.join(DATA, 'train.csv'))
df = D['df']; y = D['y']
fold = make_folds(df, 5)
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

FEAT = {}
for f in range(5):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    fb = FB3().fit(D, tri)
    FEAT[f] = (fb.transform(D, tri, loo=True), fb.transform(D, vai),
               {k: len(v) for k, v in fb.vocab.items()}, tri, vai)
log(f'features done [{time.time()-t00:.0f}s]')


def tokenize(tok, seqs):
    e = tok([' '.join(list(s)) for s in seqs], return_tensors='pt', padding='max_length',
            truncation=True, max_length=MAXLEN)
    return e['input_ids'], e['attention_mask']


@torch.no_grad()
def pooled(model, ids, mask, spans, bs=96):
    """(mean over chain, mean over CDR3 span) concatenated -> (n, 2*hidden)."""
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
        pa = (h * m).sum(1) / m.sum(1)
        pc = (h * cm).sum(1) / cm.sum(1).clamp(min=1)
        out[i:i + len(ii)] = torch.cat([pa, pc], 1).cpu().numpy()
    return out


def zs(A, rows):
    A = A.astype(np.float32); s = A[rows]
    return (A - s.mean(0, keepdims=True)) / (s.std(0, keepdims=True) + 1e-5)


# ---- frozen encoders, computed once ----
FR = {}
for nm, repo, tc in FROZEN:
    t0 = time.time()
    tok = tc.from_pretrained(repo)
    mdl = AutoModel.from_pretrained(repo).cuda().eval()
    hid, hm = tokenize(tok, heavy); lid, lm = tokenize(tok, light)
    FR[nm] = (pooled(mdl, hid, hm, H_SPAN), pooled(mdl, lid, lm, L_SPAN))
    log(f'  frozen {nm} {FR[nm][0].shape} [{time.time()-t0:.0f}s]')
    del mdl; torch.cuda.empty_cache()

FT_TOK = BertTokenizer.from_pretrained(FT_REPO)
HID_A, HM_A = tokenize(FT_TOK, heavy)
LID_A, LM_A = tokenize(FT_TOK, light)


class CrossFT(nn.Module):
    def __init__(self, repo, de=128):
        super().__init__()
        self.enc = AutoModel.from_pretrained(repo)
        h = self.enc.config.hidden_size
        self.ph = nn.Linear(h, de); self.pl = nn.Linear(h, de)
        self.bil = nn.Parameter(torch.eye(de) * 0.1)
        self.scale = nn.Parameter(torch.tensor(2.0))

    def emb(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    def forward(self, hid, hm, lid, lm):
        eh = self.ph(self.emb(hid, hm)); el = self.pl(self.emb(lid, lm))
        B = eh.shape[0] // 8
        eh = eh.reshape(B, 8, -1); el = el.reshape(B, 8, -1)
        return torch.einsum('bid,de,bje->bij', eh, self.bil, el) * self.scale


def finetune(tri, seed=0):
    """Fine-tune on the training blocks of one fold, return the adapted encoder."""
    torch.manual_seed(seed)
    o, B = block_order(df['block_id'].values[tri])
    rows = tri[o]
    hids = IDX_H[rows]; lrow = rows.reshape(B, 8)[:, 0]
    lids = np.stack([IDX_L[r] for r in lrow]).reshape(-1)
    tgt_all = torch.tensor(y[rows].reshape(B, 8))
    m = CrossFT(FT_REPO).cuda()
    opt = torch.optim.AdamW([
        {'params': m.enc.parameters(), 'lr': 2e-5},
        {'params': [p for n_, p in m.named_parameters() if not n_.startswith('enc.')], 'lr': 1e-3}],
        weight_decay=0.01)
    steps = FT_EPOCHS * ((B + FT_BLOCKS - 1) // FT_BLOCKS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps,
                                              pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler()
    g = torch.Generator().manual_seed(seed)
    for ep in range(FT_EPOCHS):
        m.train(); t0 = time.time(); perm = torch.randperm(B, generator=g); tot = 0.0; nb = 0
        for k in range(0, B, FT_BLOCKS):
            bi = perm[k:k + FT_BLOCKS]
            ri = (bi[:, None] * 8 + torch.arange(8)).reshape(-1)
            hh = torch.tensor(hids[ri.numpy()]); ll = torch.tensor(lids[ri.numpy()])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                M = m(HID_A[hh].to(DEV), HM_A[hh].to(DEV), LID_A[ll].to(DEV), LM_A[ll].to(DEV))
                tg = tgt_all[bi].to(DEV)
                loss = Fn.cross_entropy(M.reshape(-1, 8), tg.reshape(-1)) + 0.5 * Fn.cross_entropy(
                    M.transpose(1, 2).reshape(-1, 8), torch.argsort(tg, 1).reshape(-1))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item(); nb += 1
        log(f'    ft ep{ep} loss={tot/max(nb,1):.4f} [{time.time()-t0:.0f}s]')
    m.eval()
    return m.enc


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


def fold_score(raw_v, blocks, yv, T=1.25):
    o, B = block_order(blocks); inv = np.argsort(o)
    p = marginals(raw_v[o].reshape(B, 8, 8), T)
    return adjusted(credit_from_scores(np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv], yv))


INCUMBENT = [0.6026, 0.5863, 0.5562, 0.5889, 0.5345]
ARMS = {'ctrl': [], 'ft': []}
RAW = {k: np.zeros((len(df), 8), np.float32) for k in ARMS}
FT_TOK_FROZEN = None

for f in range(5):
    Ftr, Fva, voc, tri, vai = FEAT[f]
    rows_h = np.unique(IDX_H[tri]); rows_l = np.unique(IDX_L[tri].reshape(-1))

    # frozen antiberta2 for the control arm (computed once, lazily)
    if FT_TOK_FROZEN is None:
        mdl = AutoModel.from_pretrained(FT_REPO).cuda().eval()
        FT_TOK_FROZEN = (pooled(mdl, HID_A, HM_A, H_SPAN), pooled(mdl, LID_A, LM_A, L_SPAN))
        del mdl; torch.cuda.empty_cache()
        log(f'  frozen antiberta2 done [{time.time()-t00:.0f}s]')

    t0 = time.time()
    enc = finetune(tri, seed=0)
    FT_H = pooled(enc, HID_A, HM_A, H_SPAN); FT_L = pooled(enc, LID_A, LM_A, L_SPAN)
    del enc; torch.cuda.empty_cache()
    log(f'  fold {f} fine-tune+extract [{time.time()-t0:.0f}s]')

    for arm, ab in (('ctrl', FT_TOK_FROZEN), ('ft', (FT_H, FT_L))):
        EH = np.concatenate([zs(ab[0], rows_h)] + [zs(FR[n][0], rows_h) for n, _, _ in FROZEN], 1)
        EL = np.concatenate([zs(ab[1], rows_l)] + [zs(FR[n][1], rows_l) for n, _, _ in FROZEN], 1)
        acc = np.zeros((len(vai), 8))
        for s in (0, 1, 2):
            acc += fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL, IDX_H[tri], IDX_L[tri],
                       IDX_H[vai], IDX_L[vai], seed=s)
        RAW[arm][vai] = acc / 3
        sc = fold_score(RAW[arm][vai], Fva['block'], y[vai])
        ARMS[arm].append(sc)
        log(f'ARM {arm:5s} fold {f} adj={sc:.4f}   (incumbent {INCUMBENT[f]:.4f}, '
            f'delta {sc-INCUMBENT[f]:+.4f}) [{time.time()-t00:.0f}s]')
    RES['folds'] = ARMS
    json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
    for k in ARMS:
        np.save(f'{ART}/rawoof_{k}.npy', RAW[k])

hum = df['species'].values == 'human'
for arm in ARMS:
    ov = fold_score(RAW[arm], df['block_id'].values, y)
    o, B = block_order(df['block_id'].values); inv = np.argsort(o)
    p = marginals(RAW[arm][o].reshape(B, 8, 8), 1.25)
    sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
    hu = adjusted(credit_from_scores(sc[hum], y[hum]))
    RES[arm] = dict(oof=ov, human=hu, folds=ARMS[arm])
    log(f'== {arm}: OOF={ov:.4f} human={hu:.4f} folds={[round(x,4) for x in ARMS[arm]]}')
json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
log('DONE', json.dumps(RES)[:1200])

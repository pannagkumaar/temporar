"""GPU job 2.

Part A (cheap): frozen-embedding ablations with the head trained on GPU -- which LM, and does
concatenating LMs beat the best single one? All arms share folds/seeds/features; only the
embedding tower input changes. Local CPU numbers for reference (3 seeds, exact marginals):
  base 0.4858 | antiberta2 0.5269 | igbert 0.5383.

Part B (the real question): does end-to-end fine-tuning of the antibody LM with the listwise
block loss beat frozen embeddings? Probed on fold 0 only, where frozen IgBert scores 0.5700.
Promoted only if it clears frozen by more than ~0.01 on that fold.
"""
import os, sys, time, json
import numpy as np

ART = os.environ.get('ARTIFACT_ROOT', './artifacts')
DATA = os.environ.get('DATA_ROOT', './dataset_public')
PROJ = os.environ.get('PROJECT_ROOT', '.')
os.makedirs(ART, exist_ok=True)
sys.path.insert(0, os.path.join(PROJ, 'working/research'))
os.chdir(PROJ)


def log(*a):
    print(*a, flush=True)


import torch
if not torch.cuda.is_available():
    log('FATAL no cuda'); sys.exit(3)
try:
    a = torch.randn(2048, 2048, device='cuda'); (a @ a).sum().item()
    b = torch.randn(8, 512, 512, device='cuda', dtype=torch.bfloat16)
    torch.bmm(b, b).float().sum().item(); torch.cuda.synchronize()
except Exception as e:
    log('FATAL broken pod', repr(e)); sys.exit(3)
log('CUDA OK', torch.cuda.get_device_name(0))

import torch.nn as nn, torch.nn.functional as Fn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from common import load, credit_from_scores, adjusted
from feats3 import FB3
from model4 import BlockMatcher4
from model import block_order, log_sinkhorn
from permmarg import marginals
from cv3 import make_folds

DEV = 'cuda'
RESULTS = {}
t00 = time.time()

# ---------------- data + features ----------------
D = load(os.path.join(DATA, 'train.csv'))
df = D['df']; y = D['y']
fold = make_folds(df, 5)
log('data', df.shape)

FEAT = {}
for f in range(5):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    fb = FB3().fit(D, tri)
    FEAT[f] = (fb.transform(D, tri, loo=True), fb.transform(D, vai),
               {k: len(v) for k, v in fb.vocab.items()}, tri, vai)
    log(f'  features fold {f} done [{time.time()-t00:.0f}s]')

# ---------------- embeddings ----------------
heavy = sorted(set(df['heavy_chain_aa'].astype(str)))
light = sorted(set(D['aas'].reshape(-1)))
hi = {s: i for i, s in enumerate(heavy)}; li = {s: i for i, s in enumerate(light)}
IDX_H = np.array([hi[s] for s in df['heavy_chain_aa'].astype(str)], np.int64)
IDX_L = np.array([[li[s] for s in row] for row in D['aas']], np.int64)
MAXLEN = 160
MODELS = {'antiberta2': 'alchemab/antiberta2', 'igbert': 'Exscientia/IgBert',
          'esm2': 'facebook/esm2_t33_650M_UR50D'}


@torch.no_grad()
def encode(model, tok, seqs, bs=128):
    out = None
    for i in range(0, len(seqs), bs):
        enc = tok([' '.join(list(s)) for s in seqs[i:i + bs]], return_tensors='pt',
                  padding=True, truncation=True, max_length=MAXLEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
        p = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy().astype(np.float16)
        if out is None:
            out = np.zeros((len(seqs), p.shape[1]), np.float16)
        out[i:i + len(p)] = p
    return out


EMB = {}
for nm, repo in MODELS.items():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModel.from_pretrained(repo).cuda().eval()
    EMB[nm] = (encode(mdl, tok, heavy), encode(mdl, tok, light))
    log(f'  embed {nm} {EMB[nm][0].shape} {time.time()-t0:.0f}s')
    del mdl; torch.cuda.empty_cache()


def zs(A):
    A = A.astype(np.float32)
    return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-5)


# ---------------- head training ----------------
def fit(Ftr, ytr, Fva, yva, voc, EH, EL, ih_tr, il_tr, ih_va, il_va, seed=0, epochs=14,
        lr=3e-3, wd=1e-4, hid=192, d=48, pdrop=0.15, de=64, edrop=0.3, colw=0.5, nb=96):
    torch.manual_seed(seed); np.random.seed(seed)
    otr, Btr = block_order(Ftr['block']); ova, _ = block_order(Fva['block'])
    T = lambda a, o: torch.tensor(a[o], device=DEV)
    Xt, Ht, Lt, Yt = T(Ftr['X'], otr), T(Ftr['cat_h'], otr), T(Ftr['cat_l'], otr), T(ytr, otr)
    Xv, Hv, Lv = T(Fva['X'], ova), T(Fva['cat_h'], ova), T(Fva['cat_l'], ova)
    IHt = IHv = ILt = ILv = None
    if EH is not None:
        IHt, ILt, IHv, ILv = T(ih_tr, otr), T(il_tr, otr), T(ih_va, ova), T(il_va, ova)
    m = BlockMatcher4(Xt.shape[-1], voc, EH, EL, d=d, hid=hid, pdrop=pdrop, de=de,
                      edrop=edrop).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((Btr + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)
    sl = lambda t, i: (None if t is None else t[i])
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, nb):
            ri = (perm[k:k + nb][:, None] * 8 + torch.arange(8)).reshape(-1).to(DEV)
            M, _ = m(Xt[ri], Ht[ri], Lt[ri], sl(IHt, ri), sl(ILt, ri), sinkhorn_iter=6)
            tgt = Yt[ri].reshape(-1, 8)
            loss = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1)) + colw * Fn.cross_entropy(
                M.transpose(1, 2).reshape(-1, 8), torch.argsort(tgt, 1).reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        Mr, _ = m(Xv, Hv, Lv, IHv, ILv, sinkhorn_iter=30)
    return Mr.reshape(-1, 8).cpu().numpy()[np.argsort(ova)]


def score_fold(raw, blocks, yv, T=1.25):
    o, B = block_order(blocks); inv = np.argsort(o)
    p = marginals(raw[o].reshape(B, 8, 8), T)
    return adjusted(credit_from_scores(np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv], yv))


ARMS = ['', 'igbert', 'antiberta2,igbert', 'antiberta2,igbert,esm2']
for arm in ARMS:
    names = [x for x in arm.split(',') if x]
    EH = np.concatenate([zs(EMB[n][0]) for n in names], 1) if names else None
    EL = np.concatenate([zs(EMB[n][1]) for n in names], 1) if names else None
    raws = np.zeros((len(df), 8), np.float32); pf = []
    t0 = time.time()
    for f in range(5):
        Ftr, Fva, voc, tri, vai = FEAT[f]
        acc = np.zeros((len(vai), 8))
        for s in (0, 1, 2):
            acc += fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL,
                       IDX_H[tri], IDX_L[tri], IDX_H[vai], IDX_L[vai], seed=s)
        raws[vai] = acc / 3
        pf.append(score_fold(raws[vai], Fva['block'], y[vai]))
    ov = score_fold(raws, df['block_id'].values, y)
    hum = df['species'].values == 'human'
    o, B = block_order(df['block_id'].values); inv = np.argsort(o)
    p = marginals(raws[o].reshape(B, 8, 8), 1.25)
    sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
    hu = adjusted(credit_from_scores(sc[hum], y[hum]))
    key = arm or 'base'
    RESULTS[key] = dict(oof=ov, human=hu, minfold=min(pf), folds=pf,
                        proxy=0.75 * ov + 0.25 * min(pf))
    log(f'ARM {key:26s} OOF={ov:.4f} human={hu:.4f} minfold={min(pf):.4f} [{time.time()-t0:.0f}s]')
    np.save(f'{ART}/rawoof_{key.replace(",","_") or "base"}.npy', raws)
    json.dump(RESULTS, open(f'{ART}/metrics.json', 'w'), indent=2)

log('PART A DONE', time.time() - t00)

# ---------------- Part B: fine-tune probe on fold 0 ----------------
FT_REPO = os.environ.get('FT_REPO', 'Exscientia/IgBert')
FT_EPOCHS = int(os.environ.get('FT_EPOCHS', '3'))


class CrossFT(nn.Module):
    def __init__(self, repo, de=128):
        super().__init__()
        self.enc = AutoModel.from_pretrained(repo)
        self.enc.gradient_checkpointing_enable()
        h = self.enc.config.hidden_size
        self.ph = nn.Linear(h, de); self.pl = nn.Linear(h, de)
        self.bil = nn.Parameter(torch.eye(de) * 0.1)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def embed(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    def forward(self, hid, hmask, lid, lmask):
        """hid (B*8,L) heavies, lid (B*8,L) lights (block-major) -> (B,8,8)"""
        eh = self.ph(self.embed(hid, hmask))
        el = self.pl(self.embed(lid, lmask))
        B = eh.shape[0] // 8
        eh = eh.reshape(B, 8, -1); el = el.reshape(B, 8, -1)
        return torch.einsum('bid,de,bje->bij', eh, self.bil, el) * self.scale


def tokenize(tok, seqs):
    e = tok([' '.join(list(s)) for s in seqs], return_tensors='pt', padding='max_length',
            truncation=True, max_length=MAXLEN)
    return e['input_ids'], e['attention_mask']


try:
    tok = AutoTokenizer.from_pretrained(FT_REPO)
    tri = np.where(fold != 0)[0]; vai = np.where(fold == 0)[0]
    otr, Btr = block_order(df['block_id'].values[tri])
    ova, Bva = block_order(df['block_id'].values[vai])
    HSEQ = df['heavy_chain_aa'].astype(str).values
    # per block: 8 heavies (row order) and the block's 8 candidate lights (row 0's list)
    def pack(idx, order, B):
        rows = idx[order]
        hid, hm = tokenize(tok, list(HSEQ[rows]))
        lrows = rows.reshape(B, 8)[:, 0]
        lseq = [D['aas'][r, j] for r in lrows for j in range(8)]
        lid, lm = tokenize(tok, lseq)
        tgt = torch.tensor(y[rows].reshape(B, 8))
        return hid, hm, lid, lm, tgt
    Htr = pack(tri, otr, Btr); Hva = pack(vai, ova, Bva)
    log(f'  FT tokenised train {Btr} blocks val {Bva} blocks [{time.time()-t00:.0f}s]')

    torch.manual_seed(0)
    m = CrossFT(FT_REPO).cuda()
    opt = torch.optim.AdamW([
        {'params': m.enc.parameters(), 'lr': 1e-5},
        {'params': [p for n, p in m.named_parameters() if not n.startswith('enc.')], 'lr': 1e-3}],
        weight_decay=0.01)
    NB = 6
    steps = FT_EPOCHS * ((Btr + NB - 1) // NB)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5, 1e-3], total_steps=steps, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler()
    g = torch.Generator().manual_seed(0)
    ft_hist = []
    for ep in range(FT_EPOCHS):
        m.train(); t0 = time.time(); perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, NB):
            bi = perm[k:k + NB]
            ri = (bi[:, None] * 8 + torch.arange(8)).reshape(-1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                M = m(Htr[0][ri].cuda(), Htr[1][ri].cuda(), Htr[2][ri].cuda(), Htr[3][ri].cuda())
                tgt = Htr[4][bi].cuda()
                loss = Fn.cross_entropy(M.reshape(-1, 8), tgt.reshape(-1)) + 0.5 * Fn.cross_entropy(
                    M.transpose(1, 2).reshape(-1, 8), torch.argsort(tgt, 1).reshape(-1))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            if k % (NB * 100) == 0:
                log(f'    ep{ep} {k}/{Btr} loss={loss.item():.4f} [{time.time()-t0:.0f}s]')
        m.eval(); outs = []
        with torch.no_grad():
            for k in range(0, Bva, 16):
                bi = torch.arange(k, min(k + 16, Bva))
                ri = (bi[:, None] * 8 + torch.arange(8)).reshape(-1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outs.append(m(Hva[0][ri].cuda(), Hva[1][ri].cuda(), Hva[2][ri].cuda(),
                                  Hva[3][ri].cuda()).float().cpu())
        Mv = torch.cat(outs).numpy()
        raw = Mv.reshape(-1, 8)[np.argsort(ova)]
        a = score_fold(raw, df['block_id'].values[vai], y[vai])
        ft_hist.append(a)
        log(f'  FT epoch {ep} fold0 adj={a:.4f}  (frozen igbert fold0 = 0.5700) '
            f'[{time.time()-t0:.0f}s]')
        np.save(f'{ART}/ft_fold0_ep{ep}.npy', raw)
    RESULTS['finetune_fold0'] = ft_hist
except Exception as e:
    import traceback
    log('FT FAILED:', traceback.format_exc()[-2000:])
    RESULTS['finetune_fold0'] = {'error': str(e)[:400]}

json.dump(RESULTS, open(f'{ART}/metrics.json', 'w'), indent=2)
log('ALL DONE', json.dumps(RESULTS)[:1500])

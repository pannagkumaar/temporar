"""GPU job 11 -- does the model need to know how much it can trust its own clock?

A. THE FOURTH ENCODER. Every encoder added so far paid: IgBert +0.057, +AntiBERTa2 +0.011,
   +ESM-2 +0.010 OOF. antiberta2-cssp is the same architecture as AntiBERTa2 but contrastively
   pretrained against STRUCTURE, so it is the one candidate with a mechanism behind it: the
   VH/VL interface is geometry, and a structure-aware encoder may carry it. IgBert_unpaired is
   included as a near-null control (same architecture as IgBert, different corpus).
   The previous attempt stalled: model4 registers the whole embedding matrix as a per-instance
   buffer, so 8 seeds x 5 folds meant 40 copies of up to 2.2 GB. model5 takes the tables as
   forward arguments instead; they live on the GPU once per fold.

B. RESIDUE-LEVEL CROSS-CHAIN INTERACTION (formulation challenger). The incumbent pools each
   chain to one vector and interacts the two bilinearly, so it cannot express "this heavy
   residue contacts that light residue" -- which is what the VH/VL interface literature says
   drives pairing. Here each chain keeps P=24 CDR3-anchored frozen-LM residue vectors and the
   pair score adds a learned-weighted sum over all PxP residue-residue interactions.
   Distinct from the rejected per-position residue towers: those learned residue IDENTITY
   embeddings from scratch (train 0.76 / val 0.45, pure memorisation); these are frozen
   contextual vectors, so a mutated residue differs from its germline form, and only a small
   projection plus a PxP weight matrix is learned.

Incumbent: 16 seeds, embedding dropout 0.45, OOF 0.5904 / human 0.6221,
folds [0.622, 0.601, 0.571, 0.608, 0.550].
"""
import os, sys, time, json, gc
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
from feats4 import FB4
from model5 import BlockMatcher5, InterfaceMatcher
from model import block_order, log_sinkhorn
from permmarg import marginals
from cv3 import make_folds

DEV = 'cuda'; t00 = time.time(); RES = {}
MAXLEN = 152
NRES = 24                     # residue positions kept per chain for part B
SPECS = [('antiberta2', 'alchemab/antiberta2', BertTokenizer),
         ('igbert', 'Exscientia/IgBert', AutoTokenizer),
         ('esm2', 'facebook/esm2_t33_650M_UR50D', AutoTokenizer)]
RESIDUE_MODEL = None          # part B not run here
INCUMBENT = [0.622, 0.601, 0.571, 0.608, 0.550]

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


def residue_slots(spans):
    """P=24 token indices per sequence: 12 before CDR3 (FR3 end, the H91/L87 region),
    the first 6 of CDR3 and the last 6 of CDR3. Residue r sits at token r+1."""
    n = len(spans)
    out = np.zeros((n, NRES), np.int64)
    for i in range(n):
        st, en = spans[i]
        idx = ([st + 1 - k for k in range(12, 0, -1)] +
               [st + 1 + k for k in range(6)] +
               [en + 1 - k for k in range(6, 0, -1)])
        out[i] = np.clip(idx, 0, MAXLEN - 1)
    return out


H_SLOT = residue_slots(H_SPAN); L_SLOT = residue_slots(L_SPAN)


@torch.no_grad()
def encode(model, ids, mask, spans, slots, want_residues, bs=96):
    """Returns pooled (n, 2*hid) and optionally residues (n, NRES, hid) as float16."""
    hd = model.config.hidden_size
    pooled = np.zeros((len(ids), hd * 2), np.float16)
    res = np.zeros((len(ids), NRES, hd), np.float16) if want_residues else None
    ar = torch.arange(MAXLEN, device=DEV)[None, :]
    for i in range(0, len(ids), bs):
        ii = ids[i:i + bs].to(DEV); mm = mask[i:i + bs].to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(input_ids=ii, attention_mask=mm).last_hidden_state.float()
        m = mm.unsqueeze(-1).float()
        sp = torch.tensor(spans[i:i + len(ii)], device=DEV)
        cm = ((ar >= sp[:, 0:1] + 1) & (ar < sp[:, 1:2] + 1)).unsqueeze(-1).float()
        pooled[i:i + len(ii)] = torch.cat(
            [(h * m).sum(1) / m.sum(1), (h * cm).sum(1) / cm.sum(1).clamp(min=1)],
            1).half().cpu().numpy()
        if want_residues:
            sl = torch.tensor(slots[i:i + len(ii)], device=DEV)
            res[i:i + len(ii)] = torch.gather(
                h, 1, sl.unsqueeze(-1).expand(-1, -1, hd)).half().cpu().numpy()
    return pooled, res


EMB = {}
RES_H = RES_L = None
for nm, repo, tc in SPECS:
    t0 = time.time()
    try:
        tok = tc.from_pretrained(repo)
        mdl = AutoModel.from_pretrained(repo).cuda().eval()
    except Exception as e:
        log(f'  SKIP {nm} ({repo}): {type(e).__name__}: {str(e)[:180]}')
        RES[f'skip_{nm}'] = str(e)[:300]
        continue
    hid, hm = tokenize(tok, heavy); lid, lm = tokenize(tok, light)
    want = (nm == RESIDUE_MODEL)
    ph, rh = encode(mdl, hid, hm, H_SPAN, H_SLOT, want)
    pl, rl = encode(mdl, lid, lm, L_SPAN, L_SLOT, want)
    EMB[nm] = (ph, pl)
    if want:
        RES_H, RES_L = rh, rl
        log(f'  residues {nm} H{rh.shape} L{rl.shape} '
            f'({(rh.nbytes+rl.nbytes)/2**30:.2f} GB fp16)')
    del mdl, hid, hm, lid, lm
    gc.collect(); torch.cuda.empty_cache()
    log(f'  embed {nm} {ph.shape} [{time.time()-t0:.0f}s]')

FEAT = {}
FEAT4 = {}
for f in range(5):
    tri = np.where(fold != f)[0]; vai = np.where(fold == f)[0]
    fb = FB3().fit(D, tri)
    FEAT[f] = (fb.transform(D, tri, loo=True), fb.transform(D, vai),
               {k: len(v) for k, v in fb.vocab.items()}, tri, vai)
    fb4 = FB4().fit(D, tri)
    FEAT4[f] = (fb4.transform(D, tri, loo=True), fb4.transform(D, vai),
                {k: len(v) for k, v in fb4.vocab.items()}, tri, vai)
log(f'features done v3={FEAT[0][0]["X"].shape[-1]} v4={FEAT4[0][0]["X"].shape[-1]} '
    f'[{time.time()-t00:.0f}s]')


def zs_gpu(parts, rows):
    """Concatenate the named pooled tables, standardise on `rows`, return ONE GPU tensor."""
    out = []
    for A in parts:
        t = torch.tensor(A, dtype=torch.float32, device=DEV)
        s = t[rows]
        out.append((t - s.mean(0, keepdim=True)) / (s.std(0, keepdim=True) + 1e-5))
        del t
    r = torch.cat(out, 1)
    del out
    torch.cuda.empty_cache()
    return r


def fold_sc(rawv, blocks, yv, T=1.25):
    o, B = block_order(blocks); inv = np.argsort(o)
    p = marginals(rawv[o].reshape(B, 8, 8), T)
    return adjusted(credit_from_scores(np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv], yv))


def score_all(raw, T=1.25):
    o, B = block_order(df['block_id'].values); inv = np.argsort(o)
    p = marginals(raw[o].reshape(B, 8, 8), T)
    sc = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
    return adjusted(credit_from_scores(sc, y)), adjusted(credit_from_scores(sc[hum], y[hum]))


def fit(Ftr, ytr, Fva, yva, voc, EH, EL, ihtr, iltr, ihva, ilva, seed,
        RHt=None, RLt=None, epochs=14, lr=3e-3, wd=1e-4, nb=96, iface=False):
    torch.manual_seed(seed); np.random.seed(seed)
    otr, Btr = block_order(Ftr['block']); ova, _ = block_order(Fva['block'])
    T = lambda a, o: torch.tensor(a[o], device=DEV)
    Xt, Ht, Lt, Yt = T(Ftr['X'], otr), T(Ftr['cat_h'], otr), T(Ftr['cat_l'], otr), T(ytr, otr)
    Xv, Hv, Lv = T(Fva['X'], ova), T(Fva['cat_h'], ova), T(Fva['cat_l'], ova)
    IHt, ILt = T(ihtr, otr), T(iltr, otr)
    IHv, ILv = T(ihva, ova), T(ilva, ova)
    m = BlockMatcher5(Xt.shape[-1], voc, EH.shape[1]).to(DEV)
    face = InterfaceMatcher(RES_H.shape[-1], NRES).to(DEV) if iface else None
    params = list(m.parameters()) + (list(face.parameters()) if iface else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    steps = epochs * ((Btr + nb - 1) // nb)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    g = torch.Generator().manual_seed(seed)

    def iface_term(ih, il):
        """ih (B*8,) heavy ids, il (B*8,8) candidate ids, block-major -> (B,8,8).
        Rows 8b..8b+7 share a candidate list, so only B*8 distinct light chains are touched."""
        nb_ = ih.shape[0] // 8
        hidx = ih.cpu().numpy()
        lidx = il[::8].reshape(-1).cpu().numpy()          # (B*8,) unique lights in this batch
        rh = torch.tensor(RES_H[hidx], dtype=torch.float32, device=DEV).reshape(nb_, 8, NRES, -1)
        rl = torch.tensor(RES_L[lidx], dtype=torch.float32, device=DEV).reshape(nb_, 8, NRES, -1)
        return face(rh, rl)

    for ep in range(epochs):
        m.train()
        if iface:
            face.train()
        perm = torch.randperm(Btr, generator=g)
        for k in range(0, Btr, nb):
            ri = (perm[k:k + nb][:, None] * 8 + torch.arange(8)).reshape(-1).to(DEV)
            M, _ = m(Xt[ri], Ht[ri], Lt[ri], IHt[ri], ILt[ri], EH, EL, sinkhorn_iter=6)
            if iface:
                M = M + iface_term(IHt[ri], ILt[ri])
            tg = Yt[ri].reshape(-1, 8)
            loss = Fn.cross_entropy(M.reshape(-1, 8), tg.reshape(-1)) + 0.5 * Fn.cross_entropy(
                M.transpose(1, 2).reshape(-1, 8), torch.argsort(tg, 1).reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sch.step()
    m.eval()
    if iface:
        face.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(ova), 8 * 64):
            sl = slice(k, min(k + 8 * 64, len(ova)))
            Mr, _ = m(Xv[sl], Hv[sl], Lv[sl], IHv[sl], ILv[sl], EH, EL, sinkhorn_iter=30)
            if iface:
                Mr = Mr + iface_term(IHv[sl], ILv[sl])
            outs.append(Mr.reshape(-1, 8).cpu())
    out = torch.cat(outs).numpy()[np.argsort(ova)]
    del m, face, Xt, Ht, Lt, Yt, Xv, Hv, Lv, IHt, ILt, IHv, ILv, opt
    gc.collect(); torch.cuda.empty_cache()
    return out


def run(tag, names, seeds=8, iface=False, src=None):
    t0 = time.time()
    src = src if src is not None else FEAT
    raw = np.zeros((len(df), 8), np.float32); pf = []
    for f in range(5):
        Ftr, Fva, voc, tri, vai = src[f]
        rows_h = torch.tensor(np.unique(IDX_H[tri]), device=DEV)
        rows_l = torch.tensor(np.unique(IDX_L[tri].reshape(-1)), device=DEV)
        EH = zs_gpu([EMB[n][0] for n in names], rows_h)
        EL = zs_gpu([EMB[n][1] for n in names], rows_l)
        acc = np.zeros((len(vai), 8))
        for s in range(seeds):
            acc += fit(Ftr, y[tri], Fva, y[vai], voc, EH, EL, IDX_H[tri], IDX_L[tri],
                       IDX_H[vai], IDX_L[vai], seed=s, iface=iface)
        raw[vai] = acc / seeds
        pf.append(fold_sc(raw[vai], Fva['block'], y[vai]))
        log(f'  [{tag}] fold {f} adj={pf[-1]:.4f} (incumbent {INCUMBENT[f]:.3f}, '
            f'{pf[-1]-INCUMBENT[f]:+.4f}) [{time.time()-t0:.0f}s]')
        del EH, EL
        gc.collect(); torch.cuda.empty_cache()
    ov, hu = score_all(raw)
    RES[tag] = dict(oof=ov, human=hu, folds=pf, dim=None, seeds=seeds, iface=iface)
    log(f'ARM {tag:20s} OOF={ov:.4f} human={hu:.4f} d_oof={ov-0.5904:+.4f} '
        f'd_hum={hu-0.6221:+.4f} [{time.time()-t0:.0f}s]')
    json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
    np.save(f'{ART}/rawoof_{tag}.npy', raw)
    return raw



B3 = ['antiberta2', 'igbert', 'esm2']
run('ctrl_v3feats', B3, src=FEAT)
run('v4_blockgate', B3, src=FEAT4)
run('v4_blockgate_s16', B3, seeds=16, src=FEAT4)
json.dump(RES, open(f'{ART}/metrics.json', 'w'), indent=2)
log('DONE', json.dumps(RES)[:1500])

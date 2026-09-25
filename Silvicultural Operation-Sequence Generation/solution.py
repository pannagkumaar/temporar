import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda")

MAXLEN = 11
NBK = 8
W_SEQ, W_TRN, W_SCH = 0.50, 0.28, 0.22
BUCKET_MID = [0.0, 1.0, 2.0, 3.5, 6.0, 10.0, 16.5, 25.0]

MEMBERS = [
    dict(kind="tf", seed=11, d=256, layers=4, heads=4, drop=0.1, epochs=8, pw=0.0),
    dict(kind="tf", seed=22, d=256, layers=4, heads=4, drop=0.2, epochs=10, pw=0.0),
    dict(kind="tf", seed=33, d=256, layers=4, heads=4, drop=0.1, epochs=8, pw=0.0),
    dict(kind="tf", seed=44, d=256, layers=4, heads=4, drop=0.2, epochs=10, pw=0.0),
    dict(kind="tf", seed=55, d=256, layers=4, heads=4, drop=0.1, epochs=8, pw=0.0),
    dict(kind="tf", seed=66, d=256, layers=4, heads=4, drop=0.2, epochs=10, pw=0.0),
]
BATCH = 256
LR = 1e-3
WD = 0.05
N_SAMPLES = 1000
N_CAND = 300
ALPHA = 1.0
LS_ACTS = 12
LS_ITER = 8
LEN_BIAS = {2: -0.03, 3: -0.01}
SAMPLE_CHUNK = 8000
PROFILE_BATCH = 50


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class Codec:
    def __init__(self, tr):
        acts = set(tr.entry_activity.tolist())
        for pw in tr.pathway.tolist():
            for t in pw.split(" "):
                acts.add(t.split(":")[0])
        self.acts = sorted(acts)
        self.aidx = {a: i + 1 for i, a in enumerate(self.acts)}
        self.n_act = len(self.acts)
        self.zones = sorted(tr.zone.unique().tolist())
        self.elevs = sorted(tr.elev_band.unique().tolist())
        self.decs = sorted(tr.entry_decade.unique().tolist())
        self.EOS = self.n_act * NBK

    def fields(self, df):
        z = df.zone.map({v: i + 1 for i, v in enumerate(self.zones)}).fillna(0).astype(np.int64).values
        e = df.elev_band.map({v: i + 1 for i, v in enumerate(self.elevs)}).fillna(0).astype(np.int64).values
        a = df.entry_activity.map(self.aidx).fillna(0).astype(np.int64).values
        d = df.entry_decade.map({v: i + 1 for i, v in enumerate(self.decs)}).fillna(0).astype(np.int64).values
        return np.stack([z, e, a, d], 1)

    def pack(self, pws):
        n = len(pws)
        C = np.zeros((n, MAXLEN), np.int64)
        B = np.zeros((n, MAXLEN), np.int64)
        L = np.zeros(n, np.int64)
        for i, pw in enumerate(pws):
            toks = pw.split(" ")
            L[i] = len(toks)
            for j, t in enumerate(toks):
                a, b = t.split(":")
                C[i, j] = self.aidx[a]
                B[i, j] = int(b)
        return C, B, L

    def text(self, c, b):
        return " ".join(f"{self.acts[c[i] - 1]}:{b[i]}" for i in range(len(c)))


def seq_tokens(C, B, L, EOS):
    n = len(L)
    tgt = np.full((n, MAXLEN + 1), -100, np.int64)
    inp_a = np.zeros((n, MAXLEN + 1), np.int64)
    inp_b = np.full((n, MAXLEN + 1), NBK, np.int64)
    for i in range(n):
        l = L[i]
        tgt[i, :l] = (C[i, :l] - 1) * NBK + B[i, :l]
        tgt[i, l] = EOS
        inp_a[i, 1:l + 1] = C[i, :l]
        inp_b[i, 1:l + 1] = B[i, :l]
    return inp_a, inp_b, tgt


def bucket_mask(logits, inp_b, tok_bucket):
    prevb = torch.where(inp_b == NBK, torch.zeros_like(inp_b), inp_b)
    run = torch.cummax(prevb, 1).values
    bad = (tok_bucket[None, None, :] < run[:, :, None]) & (tok_bucket[None, None, :] >= 0)
    return logits.masked_fill(bad, -1e4)


def bucket_mask_last(logits, inp_b, tok_bucket):
    prevb = torch.where(inp_b == NBK, torch.zeros_like(inp_b), inp_b)
    run = prevb.max(1).values
    bad = (tok_bucket[None, :] < run[:, None]) & (tok_bucket[None, :] >= 0)
    return logits.masked_fill(bad, -1e4)


class Decoder(nn.Module):
    def __init__(self, n_act, field_sizes, d, layers, heads, drop):
        super().__init__()
        self.n_act = n_act
        self.nc = len(field_sizes)
        self.ctx = nn.ModuleList([nn.Embedding(s, d) for s in field_sizes])
        self.ctx_type = nn.Parameter(torch.randn(self.nc, d) * 0.02)
        self.act = nn.Embedding(n_act + 1, d)
        self.bkt = nn.Embedding(NBK + 1, d)
        self.pos = nn.Embedding(MAXLEN + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_act * NBK + 1)
        self.drop = nn.Dropout(drop)
        T = self.nc + MAXLEN + 1
        mask = torch.full((T, T), float("-inf"))
        mask[:, :self.nc] = 0.0
        for i in range(self.nc, T):
            mask[i, self.nc:i + 1] = 0.0
        self.register_buffer("mask", mask, persistent=False)
        bk = torch.arange(n_act * NBK + 1) % NBK
        bk[-1] = -1
        self.register_buffer("tok_bucket", bk, persistent=False)

    def forward(self, ctx, inp_a, inp_b, last=False):
        n, T = inp_a.shape
        ce = torch.stack([self.ctx[j](ctx[:, j]) for j in range(self.nc)], 1) + self.ctx_type[None]
        pos = torch.arange(T, device=inp_a.device)
        te = self.act(inp_a) + self.bkt(inp_b) + self.pos(pos)[None]
        x = self.drop(torch.cat([ce, te], 1))
        L = self.nc + T
        h = self.norm(self.enc(x, mask=self.mask[:L, :L])[:, self.nc:])
        if last:
            return bucket_mask_last(self.head(h[:, -1]), inp_b, self.tok_bucket)
        return bucket_mask(self.head(h), inp_b, self.tok_bucket)


BUCKET_MID = [0.0, 1.0, 2.0, 3.5, 6.0, 10.0, 16.5, 25.0]


def calendar_table(n_dec):
    tab = np.zeros((n_dec, NBK), np.int64)
    for d in range(n_dec):
        for b in range(NBK):
            tab[d, b] = int((1965 + 10 * d + BUCKET_MID[b] - 1960) // 5.0)
    return tab


class DecoderCal(Decoder):
    def __init__(self, n_act, field_sizes, d, layers, heads, drop, horizon, n_dec):
        super().__init__(n_act, field_sizes, d, layers, heads, drop)
        tab = torch.tensor(calendar_table(n_dec))
        self.n_cal = int(tab.max()) + 1
        self.register_buffer("cal_tab", tab, persistent=False)
        self.cal_emb = nn.Embedding(self.n_cal + 1, d)
        nn.init.zeros_(self.cal_emb.weight)
        self.head_cal = nn.Linear(d, n_act * self.n_cal)
        nn.init.zeros_(self.head_cal.weight)
        nn.init.zeros_(self.head_cal.bias)
        self.register_buffer("horizon", torch.tensor(horizon, dtype=torch.long), persistent=False)

    def forward(self, ctx, inp_a, inp_b, last=False):
        dec = ctx[:, -1]
        ctx = ctx[:, :-1]
        n, T = inp_a.shape
        ce = torch.stack([self.ctx[j](ctx[:, j]) for j in range(self.nc)], 1) + self.ctx_type[None]
        pos = torch.arange(T, device=inp_a.device)
        te = self.act(inp_a) + self.bkt(inp_b) + self.pos(pos)[None]
        cb = self.cal_tab[dec][:, None, :].expand(n, T, NBK)
        ci = torch.gather(cb, 2, inp_b.clamp(max=NBK - 1)[:, :, None])[:, :, 0] + 1
        ci = torch.where(inp_b == NBK, torch.zeros_like(ci), ci)
        te = te + self.cal_emb(ci)
        x = self.drop(torch.cat([ce, te], 1))
        L = self.nc + T
        h = self.norm(self.enc(x, mask=self.mask[:L, :L])[:, self.nc:])
        if last:
            h = h[:, -1:]
        lg = self.head(h)
        m = h.shape[1]
        hc = self.head_cal(h).view(n, m, self.n_act, self.n_cal)
        idx = self.cal_tab[dec][:, None, None, :].expand(n, m, self.n_act, NBK)
        add = torch.gather(hc, 3, idx).reshape(n, m, self.n_act * NBK)
        lg = torch.cat([lg[..., :-1] + add, lg[..., -1:]], -1)
        over = (self.tok_bucket[None, :] > self.horizon[dec][:, None]) & (self.tok_bucket[None, :] >= 0)
        lg = lg.masked_fill(over[:, None, :], -1e4)
        if last:
            return bucket_mask_last(lg[:, 0], inp_b, self.tok_bucket)
        return bucket_mask(lg, inp_b, self.tok_bucket)


def member_ctx(kind, ctx_np):
    if kind == "cal":
        return np.concatenate([ctx_np, ctx_np[:, 3:4] - 1], 1)
    return ctx_np


def profile_weights(ctx_np, beta):
    _, inv, cnt = np.unique(ctx_np, axis=0, return_inverse=True, return_counts=True)
    w = cnt[inv.reshape(-1)].astype(np.float64) ** (-beta)
    return w / w.mean()


def train_member(cfg, ctx_np, C, B, L, codec, sizes):
    seed_all(cfg["seed"])
    ctx = torch.tensor(ctx_np, device=DEVICE)
    rw = torch.tensor(profile_weights(ctx_np, cfg["pw"]), device=DEVICE, dtype=torch.float32)
    ia, ib, tg = [torch.tensor(x, device=DEVICE) for x in seq_tokens(C, B, L, codec.EOS)]
    kind = cfg["kind"]
    if kind == "cal":
        horizon = [int(B[ctx_np[:, 3] == d + 1].max()) for d in range(len(codec.decs))]
        model = DecoderCal(codec.n_act, sizes, cfg["d"], cfg["layers"], cfg["heads"], cfg["drop"], horizon, len(codec.decs)).to(DEVICE)
        ctx = torch.tensor(member_ctx(kind, ctx_np), device=DEVICE)
    else:
        model = Decoder(codec.n_act, sizes, cfg["d"], cfg["layers"], cfg["heads"], cfg["drop"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    n = len(L)
    spe = (n + BATCH - 1) // BATCH
    total = spe * cfg["epochs"]
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    g = torch.Generator(device=DEVICE)
    g.manual_seed(cfg["seed"])
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE, generator=g)
        tl = 0.0
        for s in range(spe):
            bi = perm[s * BATCH:(s + 1) * BATCH]
            lg = model(ctx[bi], ia[bi], ib[bi])
            tok_loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg[bi].reshape(-1), ignore_index=-100, reduction="none").view(tg[bi].shape)
            wt = rw[bi][:, None] * (tg[bi] != -100).float()
            loss = (tok_loss * wt).sum() / wt.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss.detach())
        print(f"member seed {cfg['seed']} epoch {ep + 1}/{cfg['epochs']} loss {tl / spe:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def sample(model, ctx, n_per, gen):
    EOS = model.n_act * NBK
    rep = ctx.repeat_interleave(n_per, 0)
    outC, outB, outL = [], [], []
    for s in range(0, len(rep), SAMPLE_CHUNK):
        cx = rep[s:s + SAMPLE_CHUNK]
        m = len(cx)
        inp_a = torch.zeros((m, MAXLEN + 1), dtype=torch.long, device=DEVICE)
        inp_b = torch.full((m, MAXLEN + 1), NBK, dtype=torch.long, device=DEVICE)
        done = torch.zeros(m, dtype=torch.bool, device=DEVICE)
        L = torch.zeros(m, dtype=torch.long, device=DEVICE)
        for t in range(MAXLEN + 1):
            lg = model(cx, inp_a[:, :t + 1], inp_b[:, :t + 1], last=True).float()
            if t < 2:
                lg[:, EOS] = -1e4
            if t >= MAXLEN:
                lg[:, :EOS] = -1e4
            tok = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen)[:, 0]
            eos = tok == EOS
            L = torch.where(eos & ~done, torch.full_like(L, t), L)
            done = done | eos
            if t < MAXLEN:
                inp_a[:, t + 1] = torch.where(done, torch.zeros_like(tok), tok // NBK + 1)
                inp_b[:, t + 1] = torch.where(done, torch.full_like(tok, NBK), tok % NBK)
        outC.append(inp_a[:, 1:].cpu())
        outB.append(inp_b[:, 1:].clamp(max=NBK - 1).cpu())
        outL.append(L.cpu())
    C = torch.cat(outC)
    B = torch.cat(outB)
    L = torch.cat(outL)
    keep = torch.arange(MAXLEN)[None] < L[:, None]
    C = torch.where(keep, C, torch.zeros_like(C))
    B = torch.where(keep, B, torch.zeros_like(B))
    return C.view(len(ctx), n_per, MAXLEN), B.view(len(ctx), n_per, MAXLEN), L.view(len(ctx), n_per)


def cdf(B, L):
    ar = torch.arange(MAXLEN, device=B.device)
    mask = ar < L[..., None]
    f = torch.stack([((B == k) & mask).sum(-1) for k in range(NBK)], -1).double() / L[..., None].double()
    return torch.cumsum(f, -1)


def raw_core(Cp, Bp, Lp, Ct, Bt, Lt):
    shape = torch.broadcast_shapes(Lp.shape, Lt.shape)
    dev = Cp.device
    lp = int(Lp.max())
    lt = int(Lt.max())
    prev = torch.zeros(shape + (lt + 1,), dtype=torch.int16, device=dev)
    for i in range(lp):
        eq = (Cp[..., i:i + 1] == Ct[..., :lt]).expand(shape + (lt,))
        cols = [torch.zeros(shape, dtype=torch.int16, device=dev)]
        for j in range(lt):
            cols.append(torch.where(eq[..., j], prev[..., j] + 1, torch.maximum(prev[..., j + 1], cols[-1])))
        prev = torch.where((i < Lp)[..., None], torch.stack(cols, -1), prev)
    lcs = torch.gather(prev, -1, Lt.expand(shape)[..., None])[..., 0].double()
    s_seq = lcs / torch.maximum(Lp, Lt).double()
    ar = torch.arange(MAXLEN - 1, device=dev)
    mp = ar < (Lp - 1)[..., None]
    mt = ar < (Lt - 1)[..., None]
    gp = torch.where(mp, Cp[..., :-1] * 1000 + Cp[..., 1:], torch.full_like(Cp[..., 1:], -1))
    gt = torch.where(mt, Ct[..., :-1] * 1000 + Ct[..., 1:], torch.full_like(Ct[..., 1:], -2))
    cntP = ((gp[..., :, None] == gp[..., None, :]) & mp[..., None, :]).sum(-1)
    inter = torch.zeros(shape, dtype=torch.float64, device=dev)
    for i in range(max(min(lp - 1, MAXLEN - 1), 0)):
        cT = (gp[..., i:i + 1] == gt).sum(-1)
        contrib = torch.minimum(cntP[..., i], cT).double() / cntP[..., i].clamp(min=1).double()
        inter = inter + torch.where(mp[..., i], contrib, torch.zeros_like(contrib))
    s_trn = inter / torch.clamp(torch.maximum(Lp - 1, Lt - 1), min=1).double()
    s_sch = torch.exp(-6.0 * ((cdf(Bp, Lp) - cdf(Bt, Lt)) ** 2).mean(-1))
    return torch.exp(W_SEQ * torch.log(s_seq.clamp(0, 1) + 1e-12) + W_TRN * torch.log(s_trn.clamp(0, 1) + 1e-12) + W_SCH * torch.log(s_sch.clamp(0, 1) + 1e-12))


def utility(cands, rC, rB, rL, w, chunk=512):
    out = []
    for s in range(0, len(cands[2]), chunk):
        M = raw_core(cands[0][s:s + chunk, None], cands[1][s:s + chunk, None], cands[2][s:s + chunk, None], rC[None], rB[None], rL[None])
        out.append(M @ w)
    return torch.cat(out)


def neighbours(c, b, acts):
    l = len(c)
    out = set()
    if l > 2:
        for i in range(l):
            out.add((tuple(c[:i] + c[i + 1:]), tuple(b[:i] + b[i + 1:])))
    if l < MAXLEN:
        for i in range(l + 1):
            lo = b[i - 1] if i > 0 else 0
            hi = b[i] if i < l else NBK - 1
            for a in acts:
                for bb in range(lo, hi + 1):
                    out.add((tuple(c[:i] + [a] + c[i:]), tuple(b[:i] + [bb] + b[i:])))
    for i in range(l):
        for a in acts:
            if a != c[i]:
                out.add((tuple(c[:i] + [a] + c[i + 1:]), tuple(b)))
        lo = b[i - 1] if i > 0 else 0
        hi = b[i + 1] if i < l - 1 else NBK - 1
        for bb in range(lo, hi + 1):
            if bb != b[i]:
                out.add((tuple(c), tuple(b[:i] + [bb] + b[i + 1:])))
    out.discard((tuple(c), tuple(b)))
    return sorted(out)


def pack_list(seqs):
    n = len(seqs)
    C = np.zeros((n, MAXLEN), np.int64)
    B = np.zeros((n, MAXLEN), np.int64)
    L = np.zeros(n, np.int64)
    for i, (c, b) in enumerate(seqs):
        C[i, :len(c)] = c
        B[i, :len(b)] = b
        L[i] = len(c)
    return torch.tensor(C, device=DEVICE), torch.tensor(B, device=DEVICE), torch.tensor(L, device=DEVICE)


def neighbours_fixed(c, b, acts):
    l = len(c)
    out = set()
    for i in range(l):
        for a in acts:
            if a != c[i]:
                out.add((tuple(c[:i] + [a] + c[i + 1:]), tuple(b)))
        lo = b[i - 1] if i > 0 else 0
        hi = b[i + 1] if i < l - 1 else NBK - 1
        for bb in range(lo, hi + 1):
            if bb != b[i]:
                out.add((tuple(c), tuple(b[:i] + [bb] + b[i + 1:])))
    return sorted(out)


def improve(c, b, cur, rC, rB, rL, wd, acts, fixed):
    for _ in range(LS_ITER):
        nb = neighbours_fixed(c, b, acts) if fixed else neighbours(c, b, acts)
        if not nb:
            break
        ut = utility(pack_list(nb), rC, rB, rL, wd)
        k = int(torch.argmax(ut))
        if float(ut[k]) <= cur + 1e-9:
            break
        cur = float(ut[k])
        c, b = list(nb[k][0]), list(nb[k][1])
    return c, b, cur


def decode_profile(C, B, L):
    key = torch.cat([C, B, L[:, None]], 1)
    u, cnt = torch.unique(key, dim=0, return_counts=True)
    order = torch.argsort(-cnt, stable=True)
    u, cnt = u[order], cnt[order]
    uC, uB, uL = u[:, :MAXLEN], u[:, MAXLEN:2 * MAXLEN], u[:, 2 * MAXLEN]
    w = cnt.double() ** ALPHA
    w = w / w.sum()
    tot = np.zeros(int(uC.max()) + 1)
    np.add.at(tot, uC.numpy().ravel(), np.repeat(w.numpy(), MAXLEN) * (np.arange(MAXLEN)[None] < uL.numpy()[:, None]).ravel())
    acts = sorted(int(a) for a in np.argsort(-tot, kind="stable")[:LS_ACTS] if a > 0 and tot[a] > 0)
    rC, rB, rL, wd = uC.to(DEVICE), uB.to(DEVICE), uL.to(DEVICE), w.to(DEVICE)
    n = min(N_CAND, len(uL))
    util = utility((rC[:n], rB[:n], rL[:n]), rC, rB, rL, wd)
    j = int(torch.argmax(util))
    c, b, cur = improve(uC[j, :uL[j]].tolist(), uB[j, :uL[j]].tolist(), float(util[j]), rC, rB, rL, wd, acts, False)
    best = (cur + LEN_BIAS.get(len(c), 0.0), c, b)
    lens = uL[:n]
    for lx in range(2, 9):
        sel = torch.nonzero(lens == lx).flatten()
        if len(sel) == 0:
            continue
        jj = int(sel[int(torch.argmax(util[sel.to(DEVICE)]))])
        c2, b2, v2 = improve(uC[jj, :lx].tolist(), uB[jj, :lx].tolist(), float(util[jj]), rC, rB, rL, wd, acts, True)
        if v2 + LEN_BIAS.get(lx, 0.0) > best[0] + 1e-12:
            best = (v2 + LEN_BIAS.get(lx, 0.0), c2, b2)
    return best[1], best[2]


def validate(sub, test):
    assert list(sub.columns) == ["row_id", "pathway"]
    assert len(sub) == len(test) and sub.row_id.is_unique and set(sub.row_id) == set(test.row_id)
    for pw in sub.pathway:
        toks = pw.split(" ")
        assert 1 <= len(toks) <= MAXLEN and pw == pw.strip() and "  " not in pw
        prev = 0
        for t in toks:
            a, b = t.split(":")
            assert len(b) == 1 and 0 <= int(b) <= 7 and int(b) >= prev
            prev = int(b)


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    te = pd.read_csv(os.path.join(data_dir, "test.csv"))
    codec = Codec(tr)
    C, B, L = codec.pack(tr.pathway.tolist())
    ctx_tr = codec.fields(tr)
    sizes = [len(codec.zones) + 1, len(codec.elevs) + 1, codec.n_act + 1, len(codec.decs) + 1]
    te_f = codec.fields(te)
    prof_keys = sorted(set(map(tuple, te_f.tolist())))
    prof_ctx = torch.tensor(np.array(prof_keys, dtype=np.int64), device=DEVICE)
    print(f"train {len(tr)} test {len(te)} test profiles {len(prof_keys)}", flush=True)
    pools = [[] for _ in prof_keys]
    for mi, cfg in enumerate(MEMBERS):
        model = train_member(cfg, ctx_tr, C, B, L, codec, sizes)
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(1000 + cfg["seed"])
        pctx = torch.tensor(member_ctx(cfg["kind"], np.array(prof_keys, dtype=np.int64)), device=DEVICE)
        for s in range(0, len(prof_keys), PROFILE_BATCH):
            Cs, Bs, Ls = sample(model, pctx[s:s + PROFILE_BATCH], N_SAMPLES, gen)
            for j in range(Cs.shape[0]):
                pools[s + j].append((Cs[j], Bs[j], Ls[j]))
        print(f"member {mi + 1}/{len(MEMBERS)} sampled", flush=True)
        del model
        torch.cuda.empty_cache()
    answers = {}
    for i, key in enumerate(prof_keys):
        Cp = torch.cat([p[0] for p in pools[i]])
        Bp = torch.cat([p[1] for p in pools[i]])
        Lp = torch.cat([p[2] for p in pools[i]])
        c, b = decode_profile(Cp, Bp, Lp)
        answers[key] = codec.text(c, b)
        if (i + 1) % 200 == 0:
            print(f"decoded {i + 1}/{len(prof_keys)}", flush=True)
    sub = pd.DataFrame({"row_id": te.row_id.values, "pathway": [answers[tuple(r)] for r in te_f.tolist()]})
    validate(sub, te)
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sub.to_csv(tmp, index=False)
    back = pd.read_csv(tmp)
    validate(back, te)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} rows {len(sub)}", flush=True)


if __name__ == "__main__":
    main()

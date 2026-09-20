import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from model import log_sinkhorn


class BlockMatcher4(nn.Module):
    """Engineered pair features + gene embeddings + optional frozen protein-LM towers.

    LM embeddings are held as a frozen lookup matrix and indexed, not materialised per pair:
    a block's 8 candidates repeat across its 8 rows, so indexing keeps the (B,8,8,1024) tensor
    from ever existing.
    """

    def __init__(self, n_feat, vocab, EH=None, EL=None, d=48, hid=192, pdrop=0.15,
                 de=64, edrop=0.3):
        super().__init__()
        self.e_hv = nn.Embedding(vocab['hv'] + 1, d)
        self.e_hj = nn.Embedding(vocab['hj'] + 1, 12)
        self.e_hd = nn.Embedding(vocab['hd'] + 1, 12)
        self.e_sp = nn.Embedding(vocab['sp'] + 1, 6)
        self.e_lv = nn.Embedding(vocab['lv'] + 1, d)
        self.e_lj = nn.Embedding(vocab['lj'] + 1, 12)
        for e in (self.e_hv, self.e_hj, self.e_hd, self.e_sp, self.e_lv, self.e_lj):
            nn.init.normal_(e.weight, 0, 0.05)
        self.use_emb = EH is not None
        extra = 0
        if self.use_emb:
            self.register_buffer('EH', torch.tensor(EH, dtype=torch.float32))
            self.register_buffer('EL', torch.tensor(EL, dtype=torch.float32))
            self.ehn = nn.LayerNorm(EH.shape[1]); self.eln = nn.LayerNorm(EL.shape[1])
            self.edrop = nn.Dropout(edrop)
            self.ph = nn.Linear(EH.shape[1], de)
            self.pl = nn.Linear(EL.shape[1], de)
            self.bil_e = nn.Parameter(torch.zeros(de, de))
            extra = de * 3
        self.bn = nn.BatchNorm1d(n_feat)
        din = n_feat + d * 3 + 12 * 3 + 6 + extra
        self.mlp = nn.Sequential(
            nn.Linear(din, hid), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid // 2, 1))
        self.bil = nn.Parameter(torch.zeros(d, d))
        self.tau = nn.Parameter(torch.tensor(0.0))

    def pair_logits(self, X, cat_h, cat_l, ih=None, il=None):
        N = X.shape[0]
        xf = self.bn(X.reshape(N * 8, -1)).reshape(N, 8, -1)
        hv = self.e_hv(cat_h[:, 0]); hj = self.e_hj(cat_h[:, 1])
        hd = self.e_hd(cat_h[:, 2]); sp = self.e_sp(cat_h[:, 3])
        lv = self.e_lv(cat_l[..., 0]); lj = self.e_lj(cat_l[..., 1])
        hvb = hv.unsqueeze(1).expand(-1, 8, -1)
        parts = [xf, hvb, lv, hvb * lv, hj.unsqueeze(1).expand(-1, 8, -1),
                 hd.unsqueeze(1).expand(-1, 8, -1), lj, sp.unsqueeze(1).expand(-1, 8, -1)]
        bil = torch.einsum('nd,de,nke->nk', hv, self.bil, lv)
        if self.use_emb:
            eh = self.ph(self.edrop(self.ehn(self.EH[ih])))          # (N,de)
            el = self.pl(self.edrop(self.eln(self.EL[il])))          # (N,8,de)
            ehb = eh.unsqueeze(1).expand(-1, 8, -1)
            parts += [ehb, el, ehb * el]
            bil = bil + torch.einsum('nd,de,nke->nk', eh, self.bil_e, el)
        return self.mlp(torch.cat(parts, -1)).squeeze(-1) + bil

    def forward(self, X, cat_h, cat_l, ih=None, il=None, sinkhorn_iter=10):
        s = self.pair_logits(X, cat_h, cat_l, ih, il)
        B = s.shape[0] // 8
        M = s.reshape(B, 8, 8)
        return M, log_sinkhorn(M, sinkhorn_iter, tau=1.0 + Fn.softplus(self.tau))

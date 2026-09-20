import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from model import log_sinkhorn


class BlockMatcher2(nn.Module):
    """Adds CDR3-anchored residue-identity towers (the VH/VL interface positions) on top of
    gene embeddings + engineered pair features."""

    def __init__(self, n_feat, vocab, nres=24, d=48, dr=24, hid=224, pdrop=0.15, use_res=True):
        super().__init__()
        self.use_res = use_res
        self.e_hv = nn.Embedding(vocab['hv'] + 1, d)
        self.e_hj = nn.Embedding(vocab['hj'] + 1, 12)
        self.e_hd = nn.Embedding(vocab['hd'] + 1, 12)
        self.e_sp = nn.Embedding(vocab['sp'] + 1, 6)
        self.e_lv = nn.Embedding(vocab['lv'] + 1, d)
        self.e_lj = nn.Embedding(vocab['lj'] + 1, 12)
        for e in (self.e_hv, self.e_hj, self.e_hd, self.e_sp, self.e_lv, self.e_lj):
            nn.init.normal_(e.weight, 0, 0.05)
        extra = 0
        if use_res:
            # position-specific residue embeddings, summed over positions
            self.r_h = nn.Parameter(torch.randn(nres, 21, dr) * 0.05)
            self.r_l = nn.Parameter(torch.randn(nres, 21, dr) * 0.05)
            self.rn_h = nn.LayerNorm(dr); self.rn_l = nn.LayerNorm(dr)
            extra = dr * 3
        self.bn = nn.BatchNorm1d(n_feat)
        din = n_feat + d * 3 + 12 * 3 + 6 + extra
        self.mlp = nn.Sequential(
            nn.Linear(din, hid), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid // 2, 1))
        self.bil = nn.Parameter(torch.zeros(d, d))
        if use_res:
            self.bil_r = nn.Parameter(torch.zeros(dr, dr))
        self.tau = nn.Parameter(torch.tensor(0.0))

    def pair_logits(self, X, cat_h, cat_l, res_h=None, res_l=None):
        N = X.shape[0]
        xf = self.bn(X.reshape(N * 8, -1)).reshape(N, 8, -1)
        hv = self.e_hv(cat_h[:, 0]); hj = self.e_hj(cat_h[:, 1])
        hd = self.e_hd(cat_h[:, 2]); sp = self.e_sp(cat_h[:, 3])
        lv = self.e_lv(cat_l[..., 0]); lj = self.e_lj(cat_l[..., 1])
        hvb = hv.unsqueeze(1).expand(-1, 8, -1)
        parts = [xf, hvb, lv, hvb * lv,
                 hj.unsqueeze(1).expand(-1, 8, -1), hd.unsqueeze(1).expand(-1, 8, -1),
                 lj, sp.unsqueeze(1).expand(-1, 8, -1)]
        bil = torch.einsum('nd,de,nke->nk', hv, self.bil, lv)
        if self.use_res:
            P = res_h.shape[1]
            rh = self.rn_h(self.r_h[torch.arange(P, device=res_h.device)[None, :], res_h].sum(1))
            rl = self.rn_l(self.r_l[torch.arange(P, device=res_l.device)[None, None, :],
                                    res_l].sum(2))
            rhb = rh.unsqueeze(1).expand(-1, 8, -1)
            parts += [rhb, rl, rhb * rl]
            bil = bil + torch.einsum('nd,de,nke->nk', rh, self.bil_r, rl)
        z = torch.cat(parts, dim=-1)
        return self.mlp(z).squeeze(-1) + bil

    def forward(self, X, cat_h, cat_l, res_h=None, res_l=None, sinkhorn_iter=10):
        s = self.pair_logits(X, cat_h, cat_l, res_h, res_l)
        B = s.shape[0] // 8
        M = s.reshape(B, 8, 8)
        return M, log_sinkhorn(M, sinkhorn_iter, tau=1.0 + Fn.softplus(self.tau))

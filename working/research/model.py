import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn


def log_sinkhorn(logits, n_iter=10, tau=1.0):
    """logits (B,8,8) -> log doubly-stochastic. Rows = heavy, cols = candidate."""
    z = logits / tau
    for _ in range(n_iter):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return z


class BlockMatcher(nn.Module):
    def __init__(self, n_feat, vocab, d=48, hid=192, pdrop=0.15):
        super().__init__()
        self.e_hv = nn.Embedding(vocab['hv'] + 1, d)
        self.e_hj = nn.Embedding(vocab['hj'] + 1, 12)
        self.e_hd = nn.Embedding(vocab['hd'] + 1, 12)
        self.e_sp = nn.Embedding(vocab['sp'] + 1, 6)
        self.e_lv = nn.Embedding(vocab['lv'] + 1, d)
        self.e_lj = nn.Embedding(vocab['lj'] + 1, 12)
        for e in (self.e_hv, self.e_lv):
            nn.init.normal_(e.weight, 0, 0.05)
        for e in (self.e_hj, self.e_hd, self.e_sp, self.e_lj):
            nn.init.normal_(e.weight, 0, 0.05)
        self.bn = nn.BatchNorm1d(n_feat)
        din = n_feat + d * 3 + 12 * 3 + 6
        self.mlp = nn.Sequential(
            nn.Linear(din, hid), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid // 2, 1))
        self.bil = nn.Parameter(torch.zeros(d, d))
        self.tau = nn.Parameter(torch.tensor(0.0))

    def pair_logits(self, X, cat_h, cat_l):
        """X (N,8,F), cat_h (N,4), cat_l (N,8,2) -> (N,8)"""
        N = X.shape[0]
        xf = self.bn(X.reshape(N * 8, -1)).reshape(N, 8, -1)
        hv = self.e_hv(cat_h[:, 0]); hj = self.e_hj(cat_h[:, 1])
        hd = self.e_hd(cat_h[:, 2]); sp = self.e_sp(cat_h[:, 3])
        lv = self.e_lv(cat_l[..., 0]); lj = self.e_lj(cat_l[..., 1])
        hvb = hv.unsqueeze(1).expand(-1, 8, -1)
        z = torch.cat([xf, hvb, lv, hvb * lv,
                       hj.unsqueeze(1).expand(-1, 8, -1),
                       hd.unsqueeze(1).expand(-1, 8, -1),
                       lj, sp.unsqueeze(1).expand(-1, 8, -1)], dim=-1)
        out = self.mlp(z).squeeze(-1)
        bil = torch.einsum('nd,de,nke->nk', hv, self.bil, lv)
        return out + bil

    def forward(self, X, cat_h, cat_l, sinkhorn_iter=10):
        """X etc. come in as whole blocks: N = B*8 rows, ordered block-major."""
        s = self.pair_logits(X, cat_h, cat_l)          # (B*8, 8)
        B = s.shape[0] // 8
        M = s.reshape(B, 8, 8)
        return M, log_sinkhorn(M, sinkhorn_iter, tau=1.0 + Fn.softplus(self.tau))


def block_order(block_ids):
    """Return an index array that groups rows block-major (8 per block), plus block count."""
    import pandas as pd
    s = pd.Series(np.arange(len(block_ids)), index=block_ids)
    order = np.concatenate([g.values for _, g in s.groupby(level=0, sort=True)])
    assert len(order) % 8 == 0
    return order, len(order) // 8

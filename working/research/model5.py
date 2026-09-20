"""Same scorer as model4.BlockMatcher4, but the frozen LM embedding tables are passed in at
forward time instead of being registered as per-instance buffers.

model4 did `register_buffer('EH', ...)`, which copies the whole embedding matrix into every
model instance. With 8 seeds x 5 folds that is 40 copies of up to 2.2 GB, which is what stalled
the five-encoder job. Here the tables live on the GPU once per fold and every seed borrows them.
"""
import torch, torch.nn as nn, torch.nn.functional as Fn
from model import log_sinkhorn


class BlockMatcher5(nn.Module):
    def __init__(self, n_feat, vocab, edim, d=48, hid=192, pdrop=0.15, de=64, edrop=0.45):
        super().__init__()
        self.e_hv = nn.Embedding(vocab['hv'] + 1, d)
        self.e_hj = nn.Embedding(vocab['hj'] + 1, 12)
        self.e_hd = nn.Embedding(vocab['hd'] + 1, 12)
        self.e_sp = nn.Embedding(vocab['sp'] + 1, 6)
        self.e_lv = nn.Embedding(vocab['lv'] + 1, d)
        self.e_lj = nn.Embedding(vocab['lj'] + 1, 12)
        for e in (self.e_hv, self.e_hj, self.e_hd, self.e_sp, self.e_lv, self.e_lj):
            nn.init.normal_(e.weight, 0, 0.05)
        self.ehn = nn.LayerNorm(edim); self.eln = nn.LayerNorm(edim)
        self.edrop = nn.Dropout(edrop)
        self.ph = nn.Linear(edim, de); self.pl = nn.Linear(edim, de)
        self.bil_e = nn.Parameter(torch.zeros(de, de))
        self.bn = nn.BatchNorm1d(n_feat)
        din = n_feat + d * 3 + 12 * 3 + 6 + de * 3
        self.mlp = nn.Sequential(
            nn.Linear(din, hid), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(hid // 2, 1))
        self.bil = nn.Parameter(torch.zeros(d, d))
        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, X, cat_h, cat_l, ih, il, EH, EL, sinkhorn_iter=10):
        N = X.shape[0]
        xf = self.bn(X.reshape(N * 8, -1)).reshape(N, 8, -1)
        hv = self.e_hv(cat_h[:, 0]); hj = self.e_hj(cat_h[:, 1])
        hd = self.e_hd(cat_h[:, 2]); sp = self.e_sp(cat_h[:, 3])
        lv = self.e_lv(cat_l[..., 0]); lj = self.e_lj(cat_l[..., 1])
        hvb = hv.unsqueeze(1).expand(-1, 8, -1)
        eh = self.ph(self.edrop(self.ehn(EH[ih])))
        el = self.pl(self.edrop(self.eln(EL[il])))
        ehb = eh.unsqueeze(1).expand(-1, 8, -1)
        z = torch.cat([xf, hvb, lv, hvb * lv,
                       hj.unsqueeze(1).expand(-1, 8, -1),
                       hd.unsqueeze(1).expand(-1, 8, -1),
                       lj, sp.unsqueeze(1).expand(-1, 8, -1),
                       ehb, el, ehb * el], dim=-1)
        s = self.mlp(z).squeeze(-1)
        s = s + torch.einsum('nd,de,nke->nk', hv, self.bil, lv)
        s = s + torch.einsum('nd,de,nke->nk', eh, self.bil_e, el)
        B = s.shape[0] // 8
        M = s.reshape(B, 8, 8)
        return M, log_sinkhorn(M, sinkhorn_iter, 1.0 + Fn.softplus(self.tau))


class InterfaceMatcher(nn.Module):
    """Formulation challenger: residue-level cross-chain interaction.

    The incumbent pools each chain to ONE vector and interacts the two vectors bilinearly, so it
    can never express "this residue of the heavy chain contacts that residue of the light chain".
    The VH/VL interface is known to be carried by specific framework positions (H39, H91, L38,
    L87 in the literature). Here each chain keeps P CDR3-anchored residue vectors from a frozen
    LM, and the pair score is a learned-weighted sum over all PxP residue-residue interactions.

    This is NOT the earlier rejected "per-position residue tower": that learned an embedding of
    residue IDENTITY from scratch, which mostly restates the V gene and memorised (train 0.76 /
    val 0.45). These are frozen contextual LM vectors, so a somatically mutated residue is
    represented differently from its germline counterpart, and only a small projection plus a
    PxP weight matrix is learned.
    """

    def __init__(self, rdim, P, de=48, pdrop=0.1):
        super().__init__()
        self.P = P
        self.rn_h = nn.LayerNorm(rdim); self.rn_l = nn.LayerNorm(rdim)
        self.drop = nn.Dropout(pdrop)
        self.qh = nn.Linear(rdim, de); self.ql = nn.Linear(rdim, de)
        self.W = nn.Parameter(torch.eye(de) * 0.1)
        self.A = nn.Parameter(torch.zeros(P, P))     # learned position-pair weights
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, RH, RL):
        """RH (B,8,P,rdim) the block's 8 heavy chains; RL (B,8,P,rdim) its 8 candidate light
        chains -> (B,8,8) score matrix.

        Both are projected BEFORE the candidate axis is expanded. Expanding first would
        materialise a (B*8, 8, P, 1024) tensor -- 576 MB per batch of 96 blocks -- for nothing.
        """
        h = self.qh(self.drop(self.rn_h(RH)))                    # (B,8,P,de)
        l = self.ql(self.drop(self.rn_l(RL)))                    # (B,8,P,de)
        hW = torch.einsum('bipd,de->bipe', h, self.W)            # (B,8,P,de)
        inter = torch.einsum('bipe,bjqe->bijpq', hW, l)          # (B,8,8,P,P)
        a = torch.softmax(self.A.reshape(-1), 0).reshape(self.P, self.P)
        return (inter * a).sum((-1, -2)) * self.scale

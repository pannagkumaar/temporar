"""Decoder comparison on saved OOF logits: raw / Sinkhorn / exact permutation marginals."""
import sys, numpy as np, pandas as pd, time
sys.path.insert(0, 'working/research')
from common import load, credit_from_scores, adjusted
from permmarg import marginals
from model import block_order
import torch
from model import log_sinkhorn

tag = sys.argv[1] if len(sys.argv) > 1 else 'v3s3'
D = load('train.csv'); df = D['df']; y = D['y']
raw = np.load(f'working/cache/oofraw_{tag}.npy')
sk = np.load(f'working/cache/oof_{tag}.npy')
o, B = block_order(df['block_id'].values)
inv = np.argsort(o)
M = raw[o].reshape(B, 8, 8)

print('raw            adj=%.4f' % adjusted(credit_from_scores(raw, y)))
print('sinkhorn(train)adj=%.4f' % adjusted(credit_from_scores(sk, y)))
for T in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    s = log_sinkhorn(torch.tensor(M), 40, tau=T).numpy().reshape(-1, 8)[inv]
    print('  sinkhorn T=%.2f adj=%.4f' % (T, adjusted(credit_from_scores(s, y))))
print('--- exact permutation marginals ---')
best = (0, None)
for T in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
    t0 = time.time()
    p = marginals(M, T)
    s = np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv]
    a = adjusted(credit_from_scores(s, y))
    print('  perm T=%.2f adj=%.4f  (%.1fs for %d blocks)' % (T, a, time.time() - t0, B))
    if a > best[0]:
        best = (a, T)
print('best perm T=%s adj=%.4f' % (best[1], best[0]))
p = marginals(M, best[1])
np.save(f'working/cache/oofperm_{tag}.npy', np.log(np.maximum(p, 1e-300)).reshape(-1, 8)[inv])

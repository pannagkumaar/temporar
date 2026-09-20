"""Exact permutation marginals for 8x8 blocks.

The model emits pairwise log-potentials M[i,j]. The bijection makes the block a permutation
model, P(pi) propto prod_i exp(M[i, pi(i)]). The metric pays (8-rank)/7, so expected credit is
maximised by ranking candidates by the true marginal P(pi(i)=j) -- not by the raw potential and
not by a Sinkhorn approximation to it.

    P(pi(i)=j) = W[i,j] * perm(W with row i and column j deleted) / perm(W)

n=8 makes this exactly computable. Ryser's formula costs 2^7*7 per 7x7 minor and there are 64
minors per block, which is nothing. Vectorised over blocks.
"""
import numpy as np

_SUB7 = None


def _subsets(n):
    """(2^n-1, n) boolean subset table (non-empty subsets) and their (-1)^(n-|S|) signs."""
    idx = np.arange(1, 1 << n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    sizes = bits.sum(1)
    signs = (-1.0) ** (n - sizes)
    return bits, signs


def perm_batch(A):
    """Permanent of each (n,n) matrix in A (B,n,n) via Ryser. n <= 8."""
    B, n, _ = A.shape
    if n == 0:
        return np.ones(B)
    bits, signs = _subsets(n)                      # (2^n-1, n)
    # rowsum[b, s, i] = sum_j in S A[b,i,j]
    rowsum = np.einsum('bij,sj->bsi', A, bits)
    prod = np.prod(rowsum, axis=2)                 # (B, 2^n-1)
    return prod @ signs


def marginals(M, temp=1.0, eps=1e-300):
    """M (B,8,8) log-potentials -> (B,8,8) marginal probabilities, rows sum to 1."""
    B, n, _ = M.shape
    Z = M / temp
    Z = Z - Z.max(axis=(1, 2), keepdims=True)
    W = np.exp(Z).astype(np.float64)
    # normalise scale so permanents stay in range: divide by geometric mean
    W = W / np.maximum(W.mean(axis=(1, 2), keepdims=True), 1e-300)
    out = np.zeros((B, n, n), dtype=np.float64)
    ri = np.arange(n)
    for i in range(n):
        keep_r = ri[ri != i]
        Wr = W[:, keep_r, :]                       # (B, n-1, n)
        for j in range(n):
            keep_c = ri[ri != j]
            out[:, i, j] = W[:, i, j] * perm_batch(np.ascontiguousarray(Wr[:, :, keep_c]))
    s = out.sum(axis=2, keepdims=True)
    return out / np.maximum(s, eps)


def rank_scores(M, temp=1.0):
    """Convenience: return log-marginals usable directly as ranking scores."""
    p = marginals(M, temp)
    return np.log(np.maximum(p, 1e-300))

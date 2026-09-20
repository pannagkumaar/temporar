"""Produce frozen antibody-LM embeddings for every heavy and light chain in train.csv.

Hypothesis under test: a pretrained antibody language model sees pairing-relevant sequence
structure that the hand-built features (germline-consensus SHM clock + gene lift tables) miss.
Deliverable: two embedding matrices per model, downloaded once, after which the decisive
comparison (base features vs base + embeddings, same 5-fold sample-grouped CV) runs on CPU
locally for free.

Rejected if adding the embeddings moves OOF adjusted by less than ~0.005.
"""
import os, sys, time, json
import numpy as np

ART = os.environ.get('ARTIFACT_ROOT', './artifacts')
DATA = os.environ.get('DATA_ROOT', './dataset_public')
os.makedirs(ART, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ---- 1. CUDA self-check before any spend on real work ----
import torch
if not torch.cuda.is_available():
    log('FATAL: no CUDA'); sys.exit(3)
try:
    a = torch.randn(2048, 2048, device='cuda')
    (a @ a).sum().item()
    b = torch.randn(8, 512, 512, device='cuda', dtype=torch.bfloat16)
    torch.bmm(b, b).float().sum().item()
    torch.cuda.synchronize()
except Exception as e:
    log('FATAL: broken CUDA pod:', repr(e)); sys.exit(3)
log('CUDA OK:', torch.cuda.get_device_name(0),
    round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')

import pandas as pd
from transformers import AutoTokenizer, AutoModel

tr = pd.read_csv(os.path.join(DATA, 'train.csv'))
heavy = sorted(set(tr['heavy_chain_aa'].astype(str)))
light = set()
for i in range(1, 9):
    light.update(tr[f'cand{i}_aa'].astype(str))
light = sorted(light)
log(f'unique heavy={len(heavy)} unique light={len(light)}')
log('len stats heavy', int(np.mean([len(s) for s in heavy])), max(len(s) for s in heavy))
log('len stats light', int(np.mean([len(s) for s in light])), max(len(s) for s in light))

MODELS = [('antiberta2', 'alchemab/antiberta2'),
          ('igbert', 'Exscientia/IgBert'),
          ('esm2', 'facebook/esm2_t33_650M_UR50D')]
MAXLEN = 160


@torch.no_grad()
def encode(model, tok, seqs, bs=128, tag=''):
    out = None
    t0 = time.time()
    for i in range(0, len(seqs), bs):
        chunk = [' '.join(list(s)) for s in seqs[i:i + bs]]
        enc = tok(chunk, return_tensors='pt', padding=True, truncation=True, max_length=MAXLEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast('cuda', dtype=torch.float16):
            h = model(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        pooled = pooled.float().cpu().numpy().astype(np.float16)
        if out is None:
            out = np.zeros((len(seqs), pooled.shape[1]), np.float16)
        out[i:i + len(pooled)] = pooled
        if i % (bs * 40) == 0:
            log(f'  {tag} {i}/{len(seqs)} {time.time()-t0:.0f}s')
    log(f'  {tag} done {len(seqs)} in {time.time()-t0:.0f}s  dim={out.shape[1]}')
    return out


meta = {}
for name, repo in MODELS:
    log(f'=== {name} ({repo}) ===')
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).cuda().eval()
    except Exception as e:
        log(f'SKIP {name}: {type(e).__name__}: {e}')
        meta[name] = {'error': f'{type(e).__name__}: {e}'}
        continue
    nparam = sum(p.numel() for p in model.parameters())
    log(f'  loaded {nparam/1e6:.0f}M params in {time.time()-t0:.0f}s')
    # tiny sanity batch first
    s = encode(model, tok, heavy[:64], 32, tag=f'{name}:probe')
    log(f'  probe norm={np.linalg.norm(s.astype(np.float32), axis=1).mean():.2f}')
    H = encode(model, tok, heavy, 128, tag=f'{name}:heavy')
    L = encode(model, tok, light, 128, tag=f'{name}:light')
    np.save(f'{ART}/{name}_heavy.npy', H)
    np.save(f'{ART}/{name}_light.npy', L)
    meta[name] = {'dim': int(H.shape[1]), 'params_m': round(nparam / 1e6, 1),
                  'n_heavy': len(heavy), 'n_light': len(light)}
    del model
    torch.cuda.empty_cache()

with open(f'{ART}/heavy_seqs.txt', 'w') as f:
    f.write('\n'.join(heavy))
with open(f'{ART}/light_seqs.txt', 'w') as f:
    f.write('\n'.join(light))
with open(f'{ART}/metrics.json', 'w') as f:
    json.dump(meta, f, indent=2)
log('META', json.dumps(meta))
log('DONE')

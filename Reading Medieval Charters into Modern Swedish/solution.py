import os
import sys
import traceback

EXIT_CODES = {
    'imports': 5,
    'cuda_init': 10,
    'find_data': 20,
    'read_data': 21,
    'tokenizer_download': 30,
    'model_download_load': 40,
    'lora_setup': 50,
    'retrieval': 55,
    'encode': 58,
    'training': 60,
    'lora_merge': 70,
    'generation': 80,
    'write_submission': 90,
}
STAGE = ['imports']


def set_stage(name):
    STAGE[0] = name
    print(f'[stage] {name} (exit code on failure: {EXIT_CODES[name]})', flush=True)


def report_failure(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    code = EXIT_CODES.get(STAGE[0], 99)
    msg = str(value).replace('\n', ' ')[:600]
    sys.stdout.flush()
    sys.stderr.write(f'FAILED stage={STAGE[0]} exit_code={code} error={etype.__name__}: {msg}\n')
    sys.stderr.flush()
    os._exit(code)


sys.excepthook = report_failure

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import csv
import json
import math
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL = 'utter-project/EuroLLM-9B-Instruct-2512'
REVISION = 'def82454026b0353f12e48a0de081a9ed380d91e'
SEED = 0
EPOCHS = 2
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SRC = 1024
MAX_TGT = 512
TOK_BUDGET = 2048
MAX_BS = 8
EX_PER_UPDATE = 16
MAX_NEW = 400
GEN_TOK_BUDGET = 20000
GEN_BS = 20
EOS_BIAS = -2.0
PHASES = (160, MAX_NEW)
SHOTS = 3
SHOT_CHARS = 300
CHECK_EVERY = 8
LOOP_NGRAM = 16
PREFILL_ROWS = 4
NAME_WEIGHT = 3.0
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
LANG = {'latin': 'latin', 'svenska': 'fornsvenska', 'tyska': 'medellågtyska', 'norska': 'fornnorska', 'danska': 'forndanska'}

torch.set_num_threads(8)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device('cuda')
WORD = re.compile(r'\w+')


def log(msg):
    print(msg, flush=True)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(self.drop(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return y + z * self.scale


def add_lora(model):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if cname in TARGETS and isinstance(child, nn.Linear):
                setattr(mod, cname, LoRALinear(child, LORA_R, LORA_ALPHA, LORA_DROPOUT))
                n += 1
    return n


def merge_lora(model):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    w = child.base.weight
                    w.copy_((w.float() + (child.lora_B @ child.lora_A) * child.scale).to(w.dtype))
                setattr(mod, cname, child.base)
                n += 1
    return n


def header(lang):
    base = lang.split(',')[0].split(' ')[0].strip('?').lower()
    return LANG.get(base, lang)


def build_prompt(tok, row, shots):
    ids = tok(row['charter'], add_special_tokens=False)['input_ids']
    if len(ids) > MAX_SRC:
        h = int(MAX_SRC * 0.75)
        ids = ids[:h] + ids[-(MAX_SRC - h):]
    text = tok.decode(ids)
    parts = ['### Liknande regester\n'] + ['- ' + sh['regest'][:SHOT_CHARS] + '\n' for sh in shots] + ['\n']
    parts.append(f"### Brev ({header(row['language'])})\n{text}\n### Regest\n")
    return ''.join(parts)


def common_forms(regests, share):
    c = {}
    for r in regests:
        for w in set(w for w in WORD.findall(r) if w[0].isupper()):
            c[w] = c.get(w, 0) + 1
    return {w for w, k in c.items() if k / len(regests) >= share}


def neighbours(query_rows, pool_rows):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=300000)
    P = vec.fit_transform([r['charter'] for r in pool_rows])
    Q = vec.transform([r['charter'] for r in query_rows])
    out = []
    for a in range(0, len(query_rows), 256):
        sim = (Q[a:a + 256] @ P.T).toarray()
        for k, q in enumerate(query_rows[a:a + 256]):
            sel = []
            for j in np.argsort(-sim[k], kind='stable'):
                pr = pool_rows[j]
                if pr['charter_id'] != q['charter_id'] and pr['issuer_id'] != q['issuer_id']:
                    sel.append(pr)
                    if len(sel) == SHOTS:
                        break
            out.append(sel)
    return out


def encode_train(tok, row, shots, bos, common):
    p = [bos] + tok(build_prompt(tok, row, shots), add_special_tokens=False)['input_ids']
    enc = tok(row['regest'], add_special_tokens=False, return_offsets_mapping=True)
    t = enc['input_ids'][:MAX_TGT] + [tok.eos_token_id]
    spans = [(m.start(), m.end()) for m in WORD.finditer(row['regest']) if m.group(0)[0].isupper() and m.group(0) not in common]
    w = [NAME_WEIGHT if any(a < e and b > s for s, e in spans) else 1.0 for a, b in enc['offset_mapping'][:MAX_TGT]] + [1.0]
    return p + t, [-100] * len(p) + t, w


def make_batches(lengths, rng):
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], rng.random()))
    batches, cur, mx = [], [], 0
    for i in order:
        m = max(mx, lengths[i])
        if cur and (m * (len(cur) + 1) > TOK_BUDGET or len(cur) >= MAX_BS):
            batches.append(cur)
            cur, m = [], lengths[i]
        cur.append(i)
        mx = m
    batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(examples, pad_id, left):
    L = max(len(e[0]) for e in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for j, (a, b) in enumerate(examples):
        s = L - len(a) if left else 0
        ids[j, s:s + len(a)] = torch.tensor(a)
        lab[j, s:s + len(b)] = torch.tensor(b)
        att[j, s:s + len(a)] = 1
    return ids, lab, att


def train(model, tok, data):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.99))
    rng = random.Random(SEED)
    lengths = [len(d[0]) for d in data]
    updates = []
    for _ in range(EPOCHS):
        cur, cnt = [], 0
        for b in make_batches(lengths, rng):
            cur.append(b)
            cnt += len(b)
            if cnt >= EX_PER_UPDATE:
                updates.append(cur)
                cur, cnt = [], 0
        if cur:
            updates.append(cur)
    total = len(updates)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    model.train()
    for step, upd in enumerate(updates):
        ntgt = sum(sum(data[i][2]) for b in upd for i in b)
        acc = 0.0
        for b in upd:
            ids, lab, att = collate([data[i][:2] for i in b], tok.pad_token_id, left=False)
            wt = torch.zeros(lab.shape, dtype=torch.float32)
            for j, i in enumerate(b):
                wt[j, len(data[i][0]) - len(data[i][2]):len(data[i][0])] = torch.tensor(data[i][2])
            ids, lab, att, wt = ids.to(DEVICE), lab.to(DEVICE), att.to(DEVICE), wt.to(DEVICE)
            L = ids.shape[1]
            keep = L - int((lab != -100).float().argmax(1).min()) + 1
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=keep)
            logits = out.logits[:, :-1].float()
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, L - keep + 1:].reshape(-1),
                                 ignore_index=-100, reduction='none')
            loss = (ce * wt[:, L - keep + 1:].reshape(-1)).sum() / ntgt
            loss.backward()
            acc += loss.item()
            del out, logits, loss
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == total - 1:
            log(f'step {step + 1}/{total} loss {acc:.4f}')


def find_loop(seq, start, seen):
    for p in range(max(start, LOOP_NGRAM), len(seq) + 1):
        g = tuple(seq[p - LOOP_NGRAM:p])
        if g in seen:
            return p - LOOP_NGRAM
        seen[g] = p
    return -1


@torch.no_grad()
def prefill_rows(model, ids, att):
    parts = []
    last = []
    for a in range(0, ids.shape[0], PREFILL_ROWS):
        c = DynamicCache()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(input_ids=ids[a:a + PREFILL_ROWS], attention_mask=att[a:a + PREFILL_ROWS], past_key_values=c,
                      use_cache=True, logits_to_keep=1)
        last.append(o.logits[:, -1, :].float())
        parts.append(c)
        del o
    cache = DynamicCache()
    for li in range(len(parts[0].layers)):
        cache.update(torch.cat([p.layers[li].keys for p in parts], 0), torch.cat([p.layers[li].values for p in parts], 0), li)
        for p in parts:
            p.layers[li].keys = None
            p.layers[li].values = None
    return cache, torch.cat(last, 0)


@torch.no_grad()
def greedy_rows(model, ids, att, max_new, eos_id, prev):
    cache, logits = prefill_rows(model, ids, att)
    n = ids.shape[0]
    buf = torch.zeros((n, max_new), dtype=torch.long, device=DEVICE)
    live = torch.arange(n, device=DEVICE)
    out = [None] * n
    ended = [False] * n
    seen = [dict() for _ in range(n)]
    checked = [0] * n
    step = 0
    while True:
        logits[:, eos_id] += EOS_BIAS
        nxt = logits.argmax(-1)
        buf[live, step] = nxt
        step += 1
        if step % CHECK_EVERY == 0 or step == max_new:
            rows = live.tolist()
            host = buf[live, :step].tolist()
            keep = []
            for k, (r, seq) in enumerate(zip(rows, host)):
                if eos_id in seq:
                    out[r] = seq[:seq.index(eos_id)]
                    ended[r] = True
                    continue
                full = prev[r] + seq
                cut = find_loop(full, checked[r], seen[r])
                checked[r] = len(full) + 1
                if cut >= 0:
                    out[r] = full[len(prev[r]):max(cut, len(prev[r]))]
                    ended[r] = True
                    continue
                if step == max_new:
                    out[r] = seq
                    continue
                keep.append(k)
            if step == max_new or not keep:
                break
            if len(keep) < len(rows):
                kt = torch.tensor(keep, device=DEVICE)
                cache.batch_select_indices(kt)
                att = att[kt]
                live = live[kt]
                nxt = nxt[kt]
        att = torch.cat([att, torch.ones((att.shape[0], 1), dtype=att.dtype, device=DEVICE)], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(input_ids=nxt[:, None], attention_mask=att, past_key_values=cache, use_cache=True)
        logits = o.logits[:, -1, :].float()
        del o
    return out, ended


def phased_greedy(model, enc, pad_id, eos_id):
    gen = [[] for _ in enc]
    pending = list(range(len(enc)))
    prev_cap = 0
    for cap in PHASES:
        step_new = cap - prev_cap
        order = sorted(pending, key=lambda i: (-(len(enc[i]) + len(gen[i])), i))
        nxt_pending = []
        k = 0
        while k < len(order):
            L = len(enc[order[k]]) + len(gen[order[k]])
            bs = max(1, min(GEN_BS, GEN_TOK_BUDGET // (L + step_new)))
            chunk = order[k:k + bs]
            k += bs
            seqs = [enc[i] + gen[i] for i in chunk]
            ids, _, att = collate([(q, q) for q in seqs], pad_id, left=True)
            toks, ended = greedy_rows(model, ids.to(DEVICE), att.to(DEVICE), step_new, eos_id, [gen[i] for i in chunk])
            for i, t, e in zip(chunk, toks, ended):
                gen[i].extend(t)
                if not e:
                    nxt_pending.append(i)
        log(f'phase {cap}: {len(order)} rows decoded, {len(nxt_pending)} continue')
        pending = nxt_pending
        prev_cap = cap
    return gen


def generate(model, tok, rows, shots, bos):
    model.eval()
    enc = [[bos] + tok(build_prompt(tok, r, sh), add_special_tokens=False)['input_ids'] for r, sh in zip(rows, shots)]
    return [tok.decode(g, skip_special_tokens=True).strip() for g in phased_greedy(model, enc, tok.pad_token_id, tok.eos_token_id)]


REQUIRED = ('train.jsonl', 'train_regests.csv', 'test.jsonl', 'sample_submission.csv')


def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    bases = ['.', 'public', 'dataset/public', 'data', 'input', os.path.join(here, 'public'),
             os.path.join(here, 'dataset', 'public'), here, '/kaggle/input']
    for base in bases:
        if all(os.path.isfile(os.path.join(base, f)) for f in REQUIRED):
            return base
    for base in ['/kaggle/input', 'dataset', 'data', 'input']:
        for root, dirs, files in os.walk(base):
            dirs.sort()
            if root.count(os.sep) - base.count(os.sep) >= 4:
                dirs[:] = []
            if all(f in files for f in REQUIRED):
                return root
    raise FileNotFoundError('public dataset directory with ' + ', '.join(REQUIRED) + ' not found')


def main():
    set_stage('cuda_init')
    torch.zeros(1, device=DEVICE)
    set_stage('find_data')
    data_dir = sys.argv[1] if len(sys.argv) > 1 else find_data_dir()
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
    log(f'data {os.path.abspath(data_dir)} -> {os.path.abspath(out_path)}')
    set_stage('read_data')
    train_rows = read_jsonl(os.path.join(data_dir, 'train.jsonl'))
    with open(os.path.join(data_dir, 'train_regests.csv'), encoding='utf-8', newline='') as f:
        reg = {r['id']: r['regest'] for r in csv.DictReader(f)}
    for r in train_rows:
        r['regest'] = reg[r['charter_id']]
    test_rows = read_jsonl(os.path.join(data_dir, 'test.jsonl'))
    with open(os.path.join(data_dir, 'sample_submission.csv'), encoding='utf-8', newline='') as f:
        sub_ids = [r['id'] for r in csv.DictReader(f)]
    log(f'train {len(train_rows)} test {len(test_rows)} submission ids {len(sub_ids)}')

    set_stage('tokenizer_download')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.pad_token = tok.eos_token
    bos = tok.bos_token_id
    set_stage('model_download_load')
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={'': 0})
    model.config.use_cache = False
    set_stage('lora_setup')
    for p in model.parameters():
        p.requires_grad_(False)
    log(f'lora modules {add_lora(model)}')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    set_stage('retrieval')
    train_shots = neighbours(train_rows, train_rows)
    test_shots = neighbours(test_rows, train_rows)
    set_stage('encode')
    train_common = common_forms([r['regest'] for r in train_rows], 0.05)
    data = [encode_train(tok, r, sh, bos, train_common) for r, sh in zip(train_rows, train_shots)]
    log(f'training examples {len(data)} mean tokens {np.mean([len(d[0]) for d in data]):.0f}')
    set_stage('training')
    train(model, tok, data)
    set_stage('lora_merge')
    model.gradient_checkpointing_disable()
    log(f'merged lora modules {merge_lora(model)}')
    model.config.use_cache = True
    torch.cuda.empty_cache()

    set_stage('generation')
    preds = generate(model, tok, test_rows, test_shots, bos)
    set_stage('write_submission')
    by_id = {r['charter_id']: p for r, p in zip(test_rows, preds)}
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(['id', 'regest'])
        for cid in sub_ids:
            w.writerow([cid, by_id.get(cid, '')])
    with open(tmp, encoding='utf-8', newline='') as f:
        back = list(csv.DictReader(f))
    assert [r['id'] for r in back] == sub_ids
    assert len(set(sub_ids)) == len(sub_ids)
    os.replace(tmp, out_path)
    log(f'wrote {out_path}: {len(back)} rows, empty {sum(1 for r in back if not r["regest"])}, '
        f'mean length {np.mean([len(r["regest"]) for r in back]):.0f}')


if __name__ == '__main__':
    main()


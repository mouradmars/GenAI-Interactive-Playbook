"""Train the Observatory's tiny GPT ("Observatory-Nano") and export it.

Everything the browser shows in the 'Inside an LLM' module is recomputed in
TypeScript from the weights exported here. Reference activations are also
exported so the TypeScript forward pass can be unit-tested against PyTorch.

Run:  python training/train.py   (≈2–4 min on a laptop CPU)
"""
import base64, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from corpus import build_corpus
from bpe import train_bpe, pretokenize, encode_word

torch.manual_seed(0); random.seed(0); np.random.seed(0)
torch.set_num_threads(2)

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "data")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- tokenizer
text = build_corpus()
chars, merges = train_bpe(text, n_merges=150)
vocab = ["<bos>"] + chars + [a + b for a, b in merges]
stoi = {t: i for i, t in enumerate(vocab)}
ranks = {m: i for i, m in enumerate(merges)}
BOS = 0

def encode(s: str) -> list[int]:
    ids = []
    for w in pretokenize(s):
        ids += [stoi[t] for t in encode_word(w, ranks)]
    return ids

sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
sent_ids = [encode(" " + s) for s in sentences]
print(f"vocab={len(vocab)}  sentences={len(sentences)}  avg tokens/sentence={np.mean([len(s) for s in sent_ids]):.1f}")

# ---------------------------------------------------------------- model
V, T, D, H, L = len(vocab), 32, 48, 4, 2
HD, M = D // H, 4 * D

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(D); s.ln2 = nn.LayerNorm(D)
        s.Wq = nn.Linear(D, D); s.Wk = nn.Linear(D, D); s.Wv = nn.Linear(D, D); s.Wo = nn.Linear(D, D)
        s.fc1 = nn.Linear(D, M); s.fc2 = nn.Linear(M, D)
    def forward(s, x, keep=None):
        B, n, _ = x.shape
        h = s.ln1(x)
        q = s.Wq(h).view(B, n, H, HD).transpose(1, 2)
        k = s.Wk(h).view(B, n, H, HD).transpose(1, 2)
        v = s.Wv(h).view(B, n, H, HD).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD)
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        if keep is not None: keep.append(att.detach())
        y = (att @ v).transpose(1, 2).reshape(B, n, D)
        x = x + s.Wo(y)
        x = x + s.fc2(F.gelu(s.fc1(s.ln2(x)), approximate="tanh"))
        return x

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, D); s.pos = nn.Embedding(T, D)
        s.blocks = nn.ModuleList([Block() for _ in range(L)])
        s.lnf = nn.LayerNorm(D); s.head = nn.Linear(D, V)
    def forward(s, idx, keep=None):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x, keep)
        return s.head(s.lnf(x))

model = GPT()
nparams = sum(p.numel() for p in model.parameters())
print("params", nparams)

def batch(bs=64):
    X = []
    for _ in range(bs):
        i = random.randrange(len(sent_ids) - 8)
        seq = [BOS]
        while len(seq) < T + 1:
            seq += sent_ids[i]; i += 1
        X.append(seq[: T + 1])
    X = torch.tensor(X)
    return X[:, :-1], X[:, 1:]

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
STEPS = 3000
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=STEPS, pct_start=0.1)
history = []
t0 = time.time()
for step in range(STEPS + 1):
    x, y = batch()
    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    if step % 25 == 0:
        history.append({"step": step, "loss": round(loss.item(), 4), "lr": round(sched.get_last_lr()[0], 6)})
    if step % 500 == 0:
        print(f"step {step:5d}  loss {loss.item():.3f}  {time.time()-t0:.0f}s")
    if step == STEPS: break
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()

model.eval()

# ---------------------------------------------------------------- export
def b64(t: torch.Tensor) -> dict:
    a = t.detach().numpy().astype("<f4")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}

W = {"tok": b64(model.tok.weight), "pos": b64(model.pos.weight),
     "lnf.g": b64(model.lnf.weight), "lnf.b": b64(model.lnf.bias),
     "head.W": b64(model.head.weight.T), "head.b": b64(model.head.bias)}
for l, b in enumerate(model.blocks):
    for name, lin in [("Wq", b.Wq), ("Wk", b.Wk), ("Wv", b.Wv), ("Wo", b.Wo), ("fc1", b.fc1), ("fc2", b.fc2)]:
        W[f"{l}.{name}.W"] = b64(lin.weight.T)   # stored as [in, out] so y = x @ W + b
        W[f"{l}.{name}.b"] = b64(lin.bias)
    for name, ln in [("ln1", b.ln1), ("ln2", b.ln2)]:
        W[f"{l}.{name}.g"] = b64(ln.weight); W[f"{l}.{name}.b"] = b64(ln.bias)

# reference activations for unit tests
refs = []
with torch.no_grad():
    for p in ["the cat likes to eat", "paris is the capital of", "three plus one is"]:
        ids = [BOS] + encode(" " + p)
        keep = []
        logits = model(torch.tensor([ids]), keep)[0, -1]
        probs = logits.softmax(-1)
        top = torch.topk(probs, 5)
        refs.append({"prompt": p, "ids": ids,
                     "lastLogits": [round(v, 5) for v in logits.tolist()],
                     "att0": [[round(v, 6) for v in row] for row in keep[0][0, 0].tolist()],
                     "top5": [[vocab[i], round(pr, 5)] for pr, i in zip(top.values.tolist(), top.indices.tolist())]})
        print(p, "→", refs[-1]["top5"][:3])

meta = {"name": "Observatory-Nano", "vocabSize": V, "ctx": T, "dModel": D, "nHeads": H, "nLayers": L,
        "dMlp": M, "params": nparams, "trainSteps": STEPS, "batchSize": 64, "optimizer": "AdamW (lr 3e-3, one-cycle, wd 0.01)",
        "corpusChars": len(text), "corpusSentences": len(sentences), "finalLoss": history[-1]["loss"]}

json.dump({"meta": meta, "vocab": vocab, "chars": chars, "merges": [list(m) for m in merges], "weights": W},
          open(os.path.join(OUT, "nano-model.json"), "w"))
sample_merges = [list(m) for m in train_bpe(text[:6000], 40)[1]]
json.dump({"refs": refs, "history": history, "sampleMerges": sample_merges}, open(os.path.join(OUT, "nano-reference.json"), "w"))
open(os.path.join(OUT, "corpus-sample.txt"), "w").write(text[:6000])
print("exported", os.path.getsize(os.path.join(OUT, "nano-model.json")) // 1024, "KB")

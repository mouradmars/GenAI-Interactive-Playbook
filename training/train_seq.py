"""Train the three tiny sequence models of module 09 on the same long-range memory task.

Task ("recall the first symbol"): a sequence x_1 … x_T of symbols from {a, b, c, d}
(all uniformly random) is read one symbol at a time; after the last symbol the model
must output x_1. The dependency distance is T − 1 steps.

Architectures (written out by hand so the TypeScript port can reproduce them exactly):
  RNN        h_t = tanh(Wx x_t + Wh h_{t−1} + b),                 logits = Wy h_T + by
  LSTM       [i f g o] = Wx x_t + Wh h_{t−1} + b;  c_t = σ(f)⊙c_{t−1} + σ(i)⊙tanh(g)
             h_t = σ(o)⊙tanh(c_t),                                logits = Wy h_T + by
  Attention  e_t = We x_t + p_t (fixed sinusoidal positions), one causal self-attention head,
             logits = Wy (o_T + e_T) + by   where o_T = Σ_s softmax(q_T·k_s/√d)_s v_s
All models: hidden width 8, identical optimizer, steps, batches and seed.

Run:  python training/train_seq.py   (≈1–2 min on CPU)  → src/data/seq-models.json
"""
import json, math, os, random
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); random.seed(0); np.random.seed(0)
torch.set_num_threads(2)
V, H, STEPS, BATCH, LR = 4, 8, 4000, 64, 5e-3
REGIMES = {"mixed": (2, 20), "long": (15, 25)}
SYMBOLS = "abcd"
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "data", "seq-models.json")


def sinusoid(T, d):
    p = np.zeros((T, d))
    for t in range(T):
        for i in range(0, d, 2):
            w = 1.0 / (10000 ** (i / d))
            p[t, i] = math.sin(t * w)
            p[t, i + 1] = math.cos(t * w)
    return torch.tensor(p, dtype=torch.float32)


class RNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.Wx = nn.Parameter(torch.randn(H, V) * 0.5); s.Wh = nn.Parameter(torch.randn(H, H) / math.sqrt(H))
        s.b = nn.Parameter(torch.zeros(H)); s.Wy = nn.Parameter(torch.randn(V, H) * 0.3); s.by = nn.Parameter(torch.zeros(V))
    def forward(s, x):  # x: [B, T, V]
        h = torch.zeros(x.shape[0], H)
        for t in range(x.shape[1]):
            h = torch.tanh(x[:, t] @ s.Wx.T + h @ s.Wh.T + s.b)
        return h @ s.Wy.T + s.by


class LSTM(nn.Module):
    def __init__(s):
        super().__init__()
        s.Wx = nn.Parameter(torch.randn(4 * H, V) * 0.5); s.Wh = nn.Parameter(torch.randn(4 * H, H) / math.sqrt(H))
        b = torch.zeros(4 * H); b[H:2 * H] = 1.0  # forget-gate bias 1 (standard practice)
        s.b = nn.Parameter(b); s.Wy = nn.Parameter(torch.randn(V, H) * 0.3); s.by = nn.Parameter(torch.zeros(V))
    def forward(s, x):
        B = x.shape[0]; h = torch.zeros(B, H); c = torch.zeros(B, H)
        for t in range(x.shape[1]):
            z = x[:, t] @ s.Wx.T + h @ s.Wh.T + s.b
            i, f, g, o = z[:, :H], z[:, H:2 * H], z[:, 2 * H:3 * H], z[:, 3 * H:]
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
        return h @ s.Wy.T + s.by


class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.We = nn.Parameter(torch.randn(H, V) * 0.5)
        s.Wq = nn.Parameter(torch.randn(H, H) / math.sqrt(H)); s.Wk = nn.Parameter(torch.randn(H, H) / math.sqrt(H))
        s.Wv = nn.Parameter(torch.randn(H, H) / math.sqrt(H))
        s.Wy = nn.Parameter(torch.randn(V, H) * 0.3); s.by = nn.Parameter(torch.zeros(V))
    def forward(s, x):
        T = x.shape[1]
        e = x @ s.We.T + sinusoid(T, H)               # [B, T, H]
        q = e[:, -1] @ s.Wq.T                          # last position only (it is the only output)
        k = e @ s.Wk.T; v = e @ s.Wv.T
        a = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(H), dim=-1)
        o = (a.unsqueeze(-1) * v).sum(1)
        return (o + e[:, -1]) @ s.Wy.T + s.by


def batch(T, n, rng):
    ids = rng.integers(0, V, size=(n, T))
    x = torch.zeros(n, T, V); x[torch.arange(n)[:, None], torch.arange(T)[None, :], torch.tensor(ids)] = 1
    return x, torch.tensor(ids[:, 0])


def train(model, name, TMIN, TMAX):
    rng = np.random.default_rng(1)  # identical data stream for every architecture
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = []
    for step in range(STEPS + 1):
        T = int(rng.integers(TMIN, TMAX + 1))
        x, y = batch(T, BATCH, rng)
        loss = nn.functional.cross_entropy(model(x), y)
        if step % 50 == 0: hist.append({"step": step, "loss": round(loss.item(), 4)})
        if step == STEPS: break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    # evaluation: fixed test sets, lengths 2 … 30 (lengths > 20 were never seen in training)
    acc = {}
    erng = np.random.default_rng(99)
    with torch.no_grad():
        for T in range(2, 31):
            x, y = batch(T, 400, erng)
            acc[T] = round((model(x).argmax(-1) == y).float().mean().item(), 4)
    print(f"{TMIN}-{TMAX} {name:5s} final loss {hist[-1]['loss']:.3f} | acc " + " ".join(f"{T}:{acc[T]:.2f}" for T in [2, 5, 10, 15, 20, 25, 30]))
    return hist, acc


def tens(t): return [[round(v, 7) for v in row] for row in t.detach().tolist()] if t.dim() == 2 else [round(v, 7) for v in t.detach().tolist()]


out = {"task": {"symbols": SYMBOLS, "regimes": REGIMES, "steps": STEPS, "batch": BATCH, "lr": LR, "hidden": H,
                "optimizer": "Adam, gradient clipping 1.0", "evaluation": "400 fixed random sequences per length, lengths 2–30"},
       "regimes": {}}
ref_seqs = ["abcd", "dabbca", "cabadbcdaabcd", "bddcacbadcbadcbaddca"]
for regime, (TMIN, TMAX) in REGIMES.items():
    out["regimes"][regime] = {}
    for name, cls in [("rnn", RNN), ("lstm", LSTM), ("attn", Attn)]:
        torch.manual_seed({"rnn": 1, "lstm": 2, "attn": 3}[name])
        m = cls()
        hist, acc = train(m, name, TMIN, TMAX)
        params = {k: tens(v) for k, v in m.named_parameters()}
        refs = []
        with torch.no_grad():
            for s in ref_seqs:
                ids = [SYMBOLS.index(ch) for ch in s]
                x = torch.zeros(1, len(ids), V); x[0, torch.arange(len(ids)), torch.tensor(ids)] = 1
                refs.append({"seq": s, "logits": [round(v, 6) for v in m(x)[0].tolist()]})
        out["regimes"][regime][name] = {"params": params, "history": hist, "accuracy": acc, "refs": refs,
                                        "nParams": sum(p.numel() for p in m.parameters())}
json.dump(out, open(OUT, "w"))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")

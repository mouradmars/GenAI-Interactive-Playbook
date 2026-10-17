"""
Train the module-04 digit classifier: 64 -> 16 -> 16 -> 10, ReLU, softmax + categorical cross-entropy.

Data: the UCI optical-recognition 8x8 digits bundled with scikit-learn (1,797 images, pixel values 0..16).
Inputs are scaled to [0, 1] (x / 16). Stratified split: 1,347 train / 450 test (random_state 0).
Training uses plain NumPy with exactly the conventions of src/engine/nn.ts (W[l] has shape n_l x n_{l-1},
mean loss over the batch, Adam with bias correction), so the browser can reproduce every number.

Output: src/data/digits.json (data, split, weights rounded to 7 decimals, training curve, reference outputs).
Run: python3 training/train_digits.py   (about 10 seconds on a laptop CPU)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 7
SIZES = [64, 16, 16, 10]
EPOCHS = 80
BATCH = 32
LR = 0.01
B1, B2, EPS = 0.9, 0.999, 1e-8

rng = np.random.default_rng(SEED)
digits = load_digits()
X_int = digits.data.astype(int)            # 1797 x 64, 0..16
y = digits.target.astype(int)
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=450, stratify=y, random_state=0)
tr, te = np.sort(tr), np.sort(te)
X = X_int / 16.0


def init():
    W, b = [], []
    for l in range(1, len(SIZES)):
        fi, fo = SIZES[l - 1], SIZES[l]
        W.append(rng.normal(0, np.sqrt(2 / fi), size=(fo, fi)))   # He normal (ReLU)
        b.append(np.zeros(fo))
    return W, b


def forward(W, b, x):
    a, zs = [x], []
    for l in range(len(W)):
        z = a[-1] @ W[l].T + b[l]
        zs.append(z)
        if l < len(W) - 1:
            a.append(np.maximum(z, 0))
        else:
            m = z.max(axis=1, keepdims=True)
            e = np.exp(z - m)
            a.append(e / e.sum(axis=1, keepdims=True))
    return a, zs


def loss_acc(W, b, ids):
    a, zs = forward(W, b, X[ids])
    z = zs[-1]
    lse = z.max(axis=1) + np.log(np.exp(z - z.max(axis=1, keepdims=True)).sum(axis=1))
    loss = float(np.mean(lse - z[np.arange(len(ids)), y[ids]]))
    acc = float(np.mean(a[-1].argmax(axis=1) == y[ids]))
    return loss, acc


W, b = init()
mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
mb = [np.zeros_like(v) for v in b]; vb = [np.zeros_like(v) for v in b]
t = 0
curve = []
l0, a0 = loss_acc(W, b, tr); lt0, at0 = loss_acc(W, b, te)
curve.append({"epoch": 0, "trainLoss": l0, "trainAcc": a0, "testLoss": lt0, "testAcc": at0})
for ep in range(1, EPOCHS + 1):
    order = rng.permutation(tr)
    for s in range(0, len(order), BATCH):
        ids = order[s:s + BATCH]
        a, zs = forward(W, b, X[ids])
        Y = np.eye(10)[y[ids]]
        delta = (a[-1] - Y) / len(ids)           # softmax + CCE, mean over batch
        gW, gb = [None] * len(W), [None] * len(W)
        for l in range(len(W) - 1, -1, -1):
            gW[l] = delta.T @ a[l]
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ W[l]) * (zs[l - 1] > 0)
        t += 1
        for l in range(len(W)):
            for P, G, M, V in ((W, gW, mW, vW), (b, gb, mb, vb)):
                M[l] = B1 * M[l] + (1 - B1) * G[l]
                V[l] = B2 * V[l] + (1 - B2) * G[l] ** 2
                mh = M[l] / (1 - B1 ** t); vh = V[l] / (1 - B2 ** t)
                P[l] = P[l] - LR * mh / (np.sqrt(vh) + EPS)
    l1, a1 = loss_acc(W, b, tr); lt, at = loss_acc(W, b, te)
    curve.append({"epoch": ep, "trainLoss": l1, "trainAcc": a1, "testLoss": lt, "testAcc": at})
    if ep % 10 == 0:
        print(f"epoch {ep:3d}  train loss {l1:.4f} acc {a1:.4f}  test loss {lt:.4f} acc {at:.4f}")

# round the exported weights and report metrics for exactly those weights
W = [np.round(w, 7) for w in W]; b = [np.round(v, 7) for v in b]
trL, trA = loss_acc(W, b, tr); teL, teA = loss_acc(W, b, te)
a, zs = forward(W, b, X[te])
pred = a[-1].argmax(axis=1)
conf = np.zeros((10, 10), dtype=int)
for t_, p_ in zip(y[te], pred):
    conf[t_, p_] += 1
ref_ids = te[:5]
ra, rz = forward(W, b, X[ref_ids])
out = {
    "source": "UCI optical recognition of handwritten digits (8x8), via sklearn.datasets.load_digits",
    "sizes": SIZES, "acts": ["relu", "relu", "softmax"], "loss": "cce",
    "training": {"optimizer": "adam", "lr": LR, "beta1": B1, "beta2": B2, "batch": BATCH, "epochs": EPOCHS, "init": "he-normal", "seed": SEED, "inputScale": "x / 16"},
    "X": X_int.tolist(), "y": y.tolist(), "train": tr.tolist(), "test": te.tolist(),
    "W": [w.tolist() for w in W], "b": [v.tolist() for v in b],
    "metrics": {"trainLoss": trL, "trainAcc": trA, "testLoss": teL, "testAcc": teA, "confusion": conf.tolist()},
    "curve": curve,
    "reference": {"ids": ref_ids.tolist(), "probs": ra[-1].round(8).tolist(), "logits": rz[-1].round(8).tolist()},
}
Path(__file__).resolve().parent.parent.joinpath("src/data/digits.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"exported: train acc {trA:.4f}, test acc {teA:.4f}, test loss {teL:.4f}")

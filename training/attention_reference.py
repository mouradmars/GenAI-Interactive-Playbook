"""
PyTorch reference outputs for module 03 (Attention).

For several deterministic configurations this script runs the attention mechanism through PyTorch in float64 and
exports inputs, parameters and intermediate tensors to src/data/attention-reference.json. The TypeScript engine
(src/engine/attention.ts) is tested against these numbers.

Two independent PyTorch paths are compared with each other before export:
  1. torch.nn.MultiheadAttention with its weights set from the row-vector parameters
     (in_proj_weight = [W_Qᵀ; W_Kᵀ; W_Vᵀ], out_proj.weight = W_Oᵀ, zero biases), per-head weights requested.
  2. Explicit per-head tensor operations with F.scaled_dot_product_attention (and a manual softmax for the
     intermediate scores), which also handle disabled scaling and head ablation.
Run: python3 training/attention_reference.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

PRESET_X = [[1.0, 0.0, 0.5, 0.2], [0.2, 1.0, 0.0, 0.4], [0.0, 0.3, 1.0, 0.0], [0.8, 0.1, 0.4, 1.0]]
PRESET_X_B = [[0.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.5, 0.0], [0.5, 0.5, 0.0, 1.0], [0.0, 0.2, 1.0, 0.3]]
PRESET = {
    "Wq": [[2, 0, 1, 0], [0, 2, 0, -1], [1, 0, 2, 0], [0, 1, 0, 2]],
    "Wk": [[1, 0, 0, 0.5], [0, 1.5, 1, 0], [-0.5, -0.5, 1, 0], [0.5, 0, -0.5, 1]],
    "Wv": [[1, 0, 0, 1], [0, 1, 0.5, 0], [0.5, 0, 1, 0], [0, 0.5, 0, 0.5]],
    "Wo": [[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]],
}


def t(x):
    return torch.tensor(x, dtype=torch.float64)


def allowed_mask(tq, tk, causal, pad, blocked):
    ok = torch.ones(tq, tk, dtype=torch.bool)
    for i in range(tq):
        for j in range(tk):
            if (causal and j > i) or pad[j] or (blocked and blocked[i][j]):
                ok[i, j] = False
    return ok


def explicit(X, Xkv, P, H, causal, scale, pad, blocked, heads_on):
    X, Xkv = t(X), t(Xkv)
    d = X.shape[1]
    dk = d // H
    ok = allowed_mask(X.shape[0], Xkv.shape[0], causal, pad, blocked)
    heads, outs = [], []
    for h in range(H):
        cs = slice(h * dk, (h + 1) * dk)
        Q, K, V = X @ t(P["Wq"])[:, cs], Xkv @ t(P["Wk"])[:, cs], Xkv @ t(P["Wv"])[:, cs]
        S = Q @ K.T
        c = 1 / np.sqrt(dk) if scale else 1.0
        Z = (S * c).masked_fill(~ok, float("-inf"))
        A = torch.softmax(Z, dim=-1)
        A = torch.nan_to_num(A, nan=0.0)  # fully masked rows: the engine's documented policy (zero weights)
        O = A @ V
        rows_ok = ok.any(dim=-1)
        if rows_ok.all():
            sdpa = F.scaled_dot_product_attention(Q[None], K[None], V[None], attn_mask=ok[None], scale=c)[0]
            assert torch.allclose(sdpa, O, atol=1e-12), "SDPA disagrees with the explicit computation"
        heads.append({"Q": Q.tolist(), "K": K.tolist(), "V": V.tolist(), "S": S.tolist(), "A": A.tolist(), "O": O.tolist()})
        outs.append(O if heads_on[h] else torch.zeros_like(O))
    concat = torch.cat(outs, dim=1)
    return heads, concat, concat @ t(P["Wo"]), ok


def via_mha(X, Xkv, P, H, ok):
    d = len(X[0])
    mha = torch.nn.MultiheadAttention(d, H, bias=True, batch_first=True, dropout=0.0)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([t(P["Wq"]).T, t(P["Wk"]).T, t(P["Wv"]).T], dim=0))
        mha.in_proj_bias.zero_()
        mha.out_proj.weight.copy_(t(P["Wo"]).T)
        mha.out_proj.bias.zero_()
    mask = torch.zeros(ok.shape).masked_fill(~ok, float("-inf"))
    out, w = mha(t(X)[None], t(Xkv)[None], t(Xkv)[None], attn_mask=mask, need_weights=True, average_attn_weights=False)
    return out[0], w[0]


def one_decimal(rng, shape, s=1.0):
    return np.round(rng.uniform(-s, s, size=shape), 1).tolist()


def case(name, X, P, H, causal=True, scale=True, pad=None, blocked=None, heads_on=None, Xkv=None):
    self_attn = Xkv is None
    Xkv = X if self_attn else Xkv
    pad = pad or [False] * len(Xkv)
    heads_on = heads_on or [True] * H
    heads, concat, out, ok = explicit(X, Xkv, P, H, causal and self_attn, scale, pad, blocked, heads_on)
    checked_mha = False
    if scale and all(heads_on) and bool(ok.any(dim=-1).all()):
        mo, mw = via_mha(X, Xkv, P, H, ok)
        assert torch.allclose(mo, out, atol=1e-12), f"{name}: MultiheadAttention output differs"
        for h in range(H):
            assert torch.allclose(mw[h], t(heads[h]["A"]), atol=1e-12), f"{name}: MultiheadAttention weights differ"
        checked_mha = True
    return {
        "name": name, "X": X, "Xkv": None if self_attn else Xkv, "params": P, "H": H, "causal": causal, "scale": scale,
        "pad": pad, "blocked": blocked, "headsOn": heads_on, "heads": heads, "concat": concat.tolist(), "out": out.tolist(),
        "checkedAgainstMultiheadAttention": checked_mha,
    }


def main():
    rng = np.random.default_rng(3)
    big = {k: one_decimal(rng, (8, 8)) for k in ["Wq", "Wk", "Wv", "Wo"]}
    Xbig = one_decimal(rng, (6, 8))
    cases = [
        case("preset · causal · h=2", PRESET_X, PRESET, 2),
        case("preset · bidirectional · h=2", PRESET_X, PRESET, 2, causal=False),
        case("preset · causal · h=1", PRESET_X, PRESET, 1),
        case("preset · causal · h=4", PRESET_X, PRESET, 4),
        case("preset · no scaling · h=2", PRESET_X, PRESET, 2, scale=False),
        case("preset · causal + padding (last key)", PRESET_X, PRESET, 2, pad=[False, False, False, True]),
        case("preset · custom knock-out", PRESET_X, PRESET, 2, causal=False, blocked=[[False] * 4, [True, False, False, False], [False] * 4, [False, True, False, True]]),
        case("preset · head 2 ablated", PRESET_X, PRESET, 2, heads_on=[True, False]),
        case("preset · fully masked first row", PRESET_X, PRESET, 2, pad=[True, False, False, False]),
        case("input B · causal · h=2", PRESET_X_B, PRESET, 2),
        case("cross-attention · 4 queries, 3 keys", PRESET_X, PRESET, 2, causal=False, Xkv=PRESET_X_B[:3]),
        case("random 6×8 · bidirectional · h=4", Xbig, big, 4, causal=False),
        case("random 6×8 · causal · h=2", Xbig, big, 2),
    ]
    # what PyTorch itself does with a fully masked row (documented in the UI)
    Q = torch.ones(1, 2, 2)
    ok = torch.tensor([[[False, False], [True, True]]])
    sdpa_full = F.scaled_dot_product_attention(Q, Q, Q, attn_mask=ok)[0, 0].tolist()
    out = {
        "torch": torch.__version__,
        "note": "float64; MultiheadAttention and F.scaled_dot_product_attention agree with the explicit computation to 1e-12",
        "fullyMaskedRowInTorchSDPA": ["nan" if np.isnan(v) else v for v in sdpa_full],
        "cases": cases,
    }
    path = Path(__file__).resolve().parent.parent / "src" / "data" / "attention-reference.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({len(cases)} cases); fully masked row in SDPA → {out['fullyMaskedRowInTorchSDPA']}")


if __name__ == "__main__":
    main()

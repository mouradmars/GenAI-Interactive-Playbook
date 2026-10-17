"""
PyTorch reference outputs for module 02 (Inside the Transformer Block).

For several deterministic configurations this script draws one-decimal parameters and inputs, runs them through
PyTorch in float64 and exports parameters plus intermediate tensors to src/data/block-reference.json.
The TypeScript engine (src/engine/block.ts) is tested against these numbers.

Two independent PyTorch paths are used:
  1. torch.nn.TransformerEncoderLayer (norm_first=True → pre-LN GPT-style block, norm_first=False → post-LN BERT-style
     block, dropout 0) and torch.nn.MultiheadAttention: PyTorch's own implementation of the whole block.
  2. Explicit tensor operations (F.layer_norm, F.scaled_dot_product_attention per head, F.gelu/silu/relu, SwiGLU) to
     obtain every intermediate tensor and to apply interventions (disabled heads, scaled sublayers, disabled units).
Run: python3 training/block_reference.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def one_decimal(rng, shape, s=1.0):
    return np.round(rng.uniform(-s, s, size=shape), 1)


def make_params(rng, d, dff):
    return {
        "ln1": {"g": np.round(rng.uniform(0.5, 1.5, d), 1), "b": one_decimal(rng, d, 0.3)},
        "ln2": {"g": np.round(rng.uniform(0.5, 1.5, d), 1), "b": one_decimal(rng, d, 0.3)},
        "Wq": one_decimal(rng, (d, d)), "Wk": one_decimal(rng, (d, d)), "Wv": one_decimal(rng, (d, d)),
        "bq": one_decimal(rng, d, 0.2), "bk": one_decimal(rng, d, 0.2), "bv": one_decimal(rng, d, 0.2),
        "Wo": one_decimal(rng, (d, d), 0.8), "bo": one_decimal(rng, d, 0.2),
        "W1": one_decimal(rng, (d, dff), 0.8), "b1": one_decimal(rng, dff, 0.2),
        "W2": one_decimal(rng, (dff, d), 0.6), "b2": one_decimal(rng, d, 0.2),
        "W3": one_decimal(rng, (d, dff), 0.8), "b3": one_decimal(rng, dff, 0.2),
    }


def t(a):
    return torch.tensor(np.asarray(a, dtype=np.float64))


def explicit(cfg, P, X, iv):
    T, d, H, dff = cfg["T"], cfg["d"], cfg["H"], cfg["dff"]
    dh = d // H
    eps = cfg["eps"]
    X = t(X)
    pre = cfg["arch"] == "pre"
    ln = lambda x, p: F.layer_norm(x, (d,), t(p["g"]), t(p["b"]), eps)
    attn_in = ln(X, P["ln1"]) if pre else X
    Q = attn_in @ t(P["Wq"]) + t(P["bq"])
    K = attn_in @ t(P["Wk"]) + t(P["bk"])
    V = attn_in @ t(P["Wv"]) + t(P["bv"])
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1) if cfg["causal"] else torch.zeros(T, T, dtype=torch.bool)
    heads, outs = [], []
    for h in range(H):
        q, k, v = Q[:, h * dh:(h + 1) * dh], K[:, h * dh:(h + 1) * dh], V[:, h * dh:(h + 1) * dh]
        S = q @ k.T
        logits = (S / np.sqrt(dh)).masked_fill(mask, float("-inf"))
        A = torch.softmax(logits, dim=-1)
        O = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=~mask[None])[0]
        assert torch.allclose(O, A @ v, atol=1e-12)
        heads.append({"S": S.tolist(), "A": A.tolist(), "O": O.tolist()})
        outs.append(O if iv["heads"][h] else torch.zeros_like(O))
    concat = torch.cat(outs, dim=1)
    attn = concat @ t(P["Wo"]) + t(P["bo"])
    s1 = X + iv["attnScale"] * attn
    Hs = s1 if pre else ln(s1, P["ln1"])
    U = ln(Hs, P["ln2"]) if pre else Hs
    Z1 = U @ t(P["W1"]) + t(P["b1"])
    um = t([1.0 if u else 0.0 for u in iv["units"]])
    if cfg["act"] == "swiglu":
        act = F.silu(Z1) * (U @ t(P["W3"]) + t(P["b3"]))
    else:
        act = {"relu": F.relu, "gelu": lambda z: F.gelu(z, approximate="none"), "silu": F.silu}[cfg["act"]](Z1)
    act = act * um
    mlp = act @ t(P["W2"]) + t(P["b2"])
    s2 = Hs + iv["mlpScale"] * mlp
    Y = s2 if pre else ln(s2, P["ln2"])
    return {"Q": Q.tolist(), "K": K.tolist(), "V": V.tolist(), "heads": heads, "concat": concat.tolist(), "attn": attn.tolist(),
            "H": Hs.tolist(), "Z1": Z1.tolist(), "act": act.tolist(), "mlp": mlp.tolist(), "Y": Y.tolist()}, attn, Y


def torch_layer(cfg, P, X):
    """The same block through torch.nn.TransformerEncoderLayer (+ MultiheadAttention)."""
    d, H, dff = cfg["d"], cfg["H"], cfg["dff"]
    act = {"relu": F.relu, "gelu": lambda z: F.gelu(z, approximate="none"), "silu": F.silu}[cfg["act"]]
    layer = torch.nn.TransformerEncoderLayer(d, H, dff, dropout=0.0, activation=act, norm_first=cfg["arch"] == "pre", batch_first=True, layer_norm_eps=cfg["eps"])
    with torch.no_grad():
        layer.self_attn.in_proj_weight.copy_(torch.cat([t(P["Wq"]).T, t(P["Wk"]).T, t(P["Wv"]).T], 0))
        layer.self_attn.in_proj_bias.copy_(torch.cat([t(P["bq"]), t(P["bk"]), t(P["bv"])]))
        layer.self_attn.out_proj.weight.copy_(t(P["Wo"]).T); layer.self_attn.out_proj.bias.copy_(t(P["bo"]))
        layer.linear1.weight.copy_(t(P["W1"]).T); layer.linear1.bias.copy_(t(P["b1"]))
        layer.linear2.weight.copy_(t(P["W2"]).T); layer.linear2.bias.copy_(t(P["b2"]))
        layer.norm1.weight.copy_(t(P["ln1"]["g"])); layer.norm1.bias.copy_(t(P["ln1"]["b"]))
        layer.norm2.weight.copy_(t(P["ln2"]["g"])); layer.norm2.bias.copy_(t(P["ln2"]["b"]))
    layer.eval()
    T = cfg["T"]
    mask = torch.nn.Transformer.generate_square_subsequent_mask(T, dtype=torch.float64) if cfg["causal"] else None
    with torch.no_grad():
        Y = layer(t(X)[None], src_mask=mask, is_causal=cfg["causal"])[0]
        xin = F.layer_norm(t(X), (d,), t(P["ln1"]["g"]), t(P["ln1"]["b"]), cfg["eps"]) if cfg["arch"] == "pre" else t(X)
        attn, _ = layer.self_attn(xin[None], xin[None], xin[None], attn_mask=mask, need_weights=False)
    return attn[0], Y


CASES = [
    {"name": "pre-causal-gelu", "cfg": {"T": 4, "d": 4, "H": 2, "dff": 8, "arch": "pre", "causal": True, "act": "gelu", "eps": 1e-5}, "seed": 1, "iv": None, "layer": True},
    {"name": "post-bidirectional-relu", "cfg": {"T": 4, "d": 4, "H": 2, "dff": 8, "arch": "post", "causal": False, "act": "relu", "eps": 1e-5}, "seed": 2, "iv": None, "layer": True},
    {"name": "pre-causal-swiglu", "cfg": {"T": 4, "d": 4, "H": 2, "dff": 8, "arch": "pre", "causal": True, "act": "swiglu", "eps": 1e-5}, "seed": 3, "iv": None, "layer": False},
    {"name": "pre-interventions-silu", "cfg": {"T": 5, "d": 4, "H": 2, "dff": 8, "arch": "pre", "causal": True, "act": "silu", "eps": 1e-5}, "seed": 4,
     "iv": {"heads": [False, True], "attnScale": 0.5, "mlpScale": 1.5, "units": [True, False, True, True, False, True, True, True]}, "layer": False},
    {"name": "pre-causal-gelu-larger", "cfg": {"T": 6, "d": 8, "H": 4, "dff": 16, "arch": "pre", "causal": True, "act": "gelu", "eps": 1e-5}, "seed": 5, "iv": None, "layer": True},
    {"name": "post-causal-gelu", "cfg": {"T": 5, "d": 6, "H": 3, "dff": 12, "arch": "post", "causal": True, "act": "gelu", "eps": 1e-5}, "seed": 6, "iv": None, "layer": True},
]

out = []
for c in CASES:
    cfg = c["cfg"]
    rng = np.random.default_rng(c["seed"])
    P = make_params(rng, cfg["d"], cfg["dff"])
    X = one_decimal(rng, (cfg["T"], cfg["d"]), 1.5)
    iv = c["iv"] or {"heads": [True] * cfg["H"], "attnScale": 1.0, "mlpScale": 1.0, "units": [True] * cfg["dff"]}
    ref, attn, Y = explicit(cfg, P, X, iv)
    if c["layer"]:
        a2, Y2 = torch_layer(cfg, P, X)
        err_a, err_y = float((a2 - attn).abs().max()), float((Y2 - Y).abs().max())
        assert err_a < 1e-12 and err_y < 1e-12, (c["name"], err_a, err_y)
        ref["torchLayerMaxErr"] = max(err_a, err_y)
    out.append({"name": c["name"], "cfg": cfg, "iv": iv, "X": X.tolist(), "params": {k: (v.tolist() if isinstance(v, np.ndarray) else {kk: vv.tolist() for kk, vv in v.items()}) for k, v in P.items()}, "ref": ref})
    print(f"{c['name']:28s} ok" + (f"  (TransformerEncoderLayer agrees to {ref['torchLayerMaxErr']:.1e})" if c["layer"] else ""))

Path(__file__).resolve().parent.parent.joinpath("src/data/block-reference.json").write_text(json.dumps({"torch": torch.__version__, "cases": out}, separators=(",", ":")))
print("wrote src/data/block-reference.json")

"""Byte-pair encoding, written to match src/engine/bpe.ts exactly.

Pre-tokenisation: a word keeps its leading space, which is rendered as '▁'.
Merges are chosen by highest pair frequency; ties are broken by the
lexicographically smallest pair so that training is fully deterministic.
"""
import re
from collections import Counter

PRE = re.compile(r" ?[a-z]+| ?[^a-z ]")
SPACE = "▁"


def pretokenize(text: str) -> list[str]:
    return [p.replace(" ", SPACE) for p in PRE.findall(text.lower())]


def train_bpe(text: str, n_merges: int):
    words = Counter(pretokenize(text))
    seqs = {w: list(w) for w in words}
    chars = sorted({c for w in words for c in w})
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, f in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        top = max(pairs.values())
        cands = sorted(p for p, c in pairs.items() if c == top)
        best = cands[0]
        merges.append(best)
        a, b = best
        for w in seqs:
            s = seqs[w]
            out, i = [], 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    return chars, merges


def encode_word(word: str, ranks: dict) -> list[str]:
    s = list(word)
    while len(s) > 1:
        best, best_i = None, -1
        for i in range(len(s) - 1):
            r = ranks.get((s[i], s[i + 1]))
            if r is not None and (best is None or r < best):
                best, best_i = r, i
        if best is None:
            break
        s = s[:best_i] + [s[best_i] + s[best_i + 1]] + s[best_i + 2:]
    return s

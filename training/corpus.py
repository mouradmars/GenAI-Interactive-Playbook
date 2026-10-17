"""Deterministic toy corpus for the Observatory's tiny language model.

The grammar is deliberately small so that a ~70k-parameter transformer can
learn real, inspectable regularities (facts, agreement, counting) that make
good classroom examples.
"""
import random

ANIMAL_FOOD = {
    "cat": "fish", "dog": "bones", "rabbit": "carrots", "bird": "seeds",
    "mouse": "cheese", "bear": "honey", "horse": "hay", "monkey": "bananas",
}
CAPITALS = {
    "paris": "france", "rome": "italy", "berlin": "germany", "madrid": "spain",
    "tokyo": "japan", "cairo": "egypt", "riyadh": "saudi arabia", "lima": "peru",
    "ottawa": "canada", "athens": "greece",
}
COLORS = {"sky": "blue", "grass": "green", "snow": "white", "sun": "yellow",
          "night": "dark", "sea": "blue", "fire": "red", "coal": "black"}
NUMS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
NAMES = ["sara", "omar", "lina", "adam", "maya", "yusuf", "nora", "sam"]
PLACES = ["park", "garden", "river", "forest", "market", "school", "beach"]
ADJS = ["small", "big", "old", "young", "happy", "sleepy", "brown", "little"]
VERBS_PAST = ["saw", "found", "fed", "followed", "watched"]


def sentence(rng: random.Random) -> str:
    k = rng.random()
    if k < 0.22:
        a = rng.choice(list(ANIMAL_FOOD))
        adj = rng.choice(ADJS) + " " if rng.random() < 0.5 else ""
        return f"the {adj}{a} likes to eat {ANIMAL_FOOD[a]}."
    if k < 0.40:
        c = rng.choice(list(CAPITALS))
        if rng.random() < 0.7:
            return f"{c} is the capital of {CAPITALS[c]}."
        return f"the capital of {CAPITALS[c]} is {c}."
    if k < 0.52:
        t = rng.choice(list(COLORS))
        return f"the {t} is {COLORS[t]}."
    if k < 0.66:
        i = rng.randrange(0, 10)
        return f"{NUMS[i]} plus one is {NUMS[i + 1]}."
    if k < 0.84:
        n = rng.choice(NAMES)
        p = rng.choice(PLACES)
        a = rng.choice(list(ANIMAL_FOOD))
        v = rng.choice(VERBS_PAST)
        return f"{n} went to the {p} and {v} a {a}."
    a = rng.choice(list(ANIMAL_FOOD))
    p = rng.choice(PLACES)
    return f"the {a} sat in the {p}."


def build_corpus(n_sentences: int = 12000, seed: int = 7) -> str:
    rng = random.Random(seed)
    return " ".join(sentence(rng) for _ in range(n_sentences))


if __name__ == "__main__":
    c = build_corpus(20)
    print(c)

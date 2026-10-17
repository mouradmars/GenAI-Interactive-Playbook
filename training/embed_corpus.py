"""Deterministic corpus for the embedding lab (module 06).

Unlike the module-01 corpus, every word here has its own profile of attributes
(habitat, food, sound, size, language, continent, gender, ...), so co-occurrence
vectors show graded, overlapping similarity instead of identical clusters.

Run:  python training/embed_corpus.py   → src/data/embed-corpus.txt
"""
import os
import random

ANIMALS = {
    # name: (habitat, foods, sound, size, legs-or-move)
    "cat": ("house", ["fish", "mice"], "meows", "small", "walks"),
    "dog": ("house", ["bones", "meat"], "barks", "medium", "runs"),
    "cow": ("farm", ["grass", "hay"], "moos", "big", "walks"),
    "horse": ("farm", ["hay", "apples"], "neighs", "big", "gallops"),
    "sheep": ("farm", ["grass"], "bleats", "medium", "walks"),
    "pig": ("farm", ["apples", "corn"], "oinks", "medium", "walks"),
    "chicken": ("farm", ["corn", "seeds"], "clucks", "small", "walks"),
    "wolf": ("forest", ["meat"], "howls", "medium", "runs"),
    "bear": ("forest", ["honey", "fish"], "growls", "big", "walks"),
    "fox": ("forest", ["mice", "meat"], "barks", "small", "runs"),
    "owl": ("forest", ["mice"], "hoots", "small", "flies"),
    "whale": ("sea", ["fish"], "sings", "huge", "swims"),
    "shark": ("sea", ["fish", "meat"], "is silent", "big", "swims"),
    "dolphin": ("sea", ["fish"], "clicks", "medium", "swims"),
    "eagle": ("mountains", ["fish", "mice"], "screams", "medium", "flies"),
    "monkey": ("jungle", ["bananas", "fruit"], "chatters", "small", "climbs"),
    "tiger": ("jungle", ["meat"], "roars", "big", "runs"),
    "parrot": ("jungle", ["seeds", "fruit"], "talks", "small", "flies"),
}

COUNTRIES = {
    # country: (capital, language, continent)
    "france": ("paris", "french", "europe"),
    "italy": ("rome", "italian", "europe"),
    "spain": ("madrid", "spanish", "europe"),
    "germany": ("berlin", "german", "europe"),
    "greece": ("athens", "greek", "europe"),
    "japan": ("tokyo", "japanese", "asia"),
    "china": ("beijing", "chinese", "asia"),
    "india": ("delhi", "hindi", "asia"),
    "egypt": ("cairo", "arabic", "africa"),
    "morocco": ("rabat", "arabic", "africa"),
    "kenya": ("nairobi", "swahili", "africa"),
    "brazil": ("brasilia", "portuguese", "america"),
    "peru": ("lima", "spanish", "america"),
    "chile": ("santiago", "spanish", "america"),
    "canada": ("ottawa", "english", "america"),
}

FRUIT = {  # fruit: (colour, taste)
    "apple": ("red", "sweet"), "banana": ("yellow", "sweet"), "lemon": ("yellow", "sour"),
    "orange": ("orange", "sweet"), "grape": ("purple", "sweet"), "lime": ("green", "sour"),
    "cherry": ("red", "sweet"), "plum": ("purple", "sour"),
}

PEOPLE = {  # word: (gender, age, royal)
    "king": ("he", "adult", True), "queen": ("she", "adult", True),
    "prince": ("he", "young", True), "princess": ("she", "young", True),
    "man": ("he", "adult", False), "woman": ("she", "adult", False),
    "boy": ("he", "young", False), "girl": ("she", "young", False),
}

JOBS = {  # job: (place, tool, object)
    "doctor": ("hospital", "medicine", "patients"), "nurse": ("hospital", "medicine", "patients"),
    "teacher": ("school", "books", "students"), "chef": ("kitchen", "knife", "food"),
    "farmer": ("farm", "tractor", "crops"), "pilot": ("airport", "plane", "passengers"),
}

NUMS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
THINGS = {"sky": "blue", "grass": "green", "snow": "white", "sun": "yellow", "coal": "black", "blood": "red", "sea": "blue"}


def sentences(rng: random.Random):
    a = rng.choice(list(ANIMALS))
    hab, foods, sound, size, move = ANIMALS[a]
    c = rng.choice(list(COUNTRIES))
    cap, lang, cont = COUNTRIES[c]
    f = rng.choice(list(FRUIT))
    col, taste = FRUIT[f]
    p = rng.choice(list(PEOPLE))
    g, age, royal = PEOPLE[p]
    j = rng.choice(list(JOBS))
    place, tool, obj = JOBS[j]
    k = rng.randrange(0, 10)
    t = rng.choice(list(THINGS))
    poss = "his" if g == "he" else "her"
    return [
        # animals
        f"the {a} lives in the {hab}", f"the {a} eats {rng.choice(foods)}", f"the {a} {sound} at night",
        f"a {size} {a} {move} quickly", f"we saw a {a} in the {hab}",
        # geography
        f"{cap} is the capital of {c}", f"we visited {cap} in {c}", f"in {cap} people speak {lang}",
        f"people in {c} speak {lang}", f"{c} is a country in {cont}", f"{cap} is a city in {cont}",
        f"she flew to {cap} last summer",
        # fruit
        f"the {f} is {col} and {taste}", f"she ate a {taste} {f}", f"a {col} {f} fell from the tree",
        # people and royalty
        (f"the {p} wore {poss} crown in the palace" if royal else f"the {p} wore {poss} coat in the street"),
        f"{g} is a {p}", f"the {age} {p} smiled and {g} waved",
        (f"the {p} ruled the kingdom with {poss} family" if royal else f"the {p} walked to work with {poss} family"),
        # jobs
        f"the {j} works in the {place}", f"the {j} uses a {tool} to help {obj}", f"{g} is a {j} at the {place}",
        # numbers and colours
        f"{NUMS[k]} plus one is {NUMS[k + 1]}", f"she counted from {NUMS[k]} to {NUMS[min(10, k + 2)]}",
        f"the {t} is {THINGS[t]}",
    ]


def build(n: int = 5000, seed: int = 2026) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        out.append(rng.choice(sentences(rng)))
    return ". ".join(out) + "."


if __name__ == "__main__":
    text = build()
    path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "embed-corpus.txt")
    open(path, "w").write(text)
    print(len(text), "chars,", text.count("."), "sentences ->", path)

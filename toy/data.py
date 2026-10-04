"""Synthetic entities, fact sheets, and topic-Markov dialogue traces (all seeded)."""
import random

ENTITIES = ["Veltrax", "Quorvane", "Ilmeth", "Draskel", "Pyrenna", "Toskarel", "Bramwick", "Seluun"]

ATTRIBUTES = [
    "founding year", "headquarters city", "chief executive", "flagship product", "employee count",
    "corporate color", "mascot animal", "motto word", "stock ticker", "primary material",
    "chief engineer", "research lab city", "annual revenue", "archive codename", "founding river",
    "lucky number", "signature dish", "anthem composer", "satellite name", "board chair",
]

COLORS = ["crimson", "teal", "ochre", "violet", "amber", "indigo", "olive", "magenta", "cobalt", "saffron"]
ANIMALS = ["otter", "heron", "lynx", "ibex", "marten", "condor", "gecko", "bison", "jackal", "puffin"]

FILLER = [
    "{e} is often discussed in regional trade journals.",
    "Analysts describe {e} as a steady but unusual organisation.",
    "The history of {e} includes several quiet reorganisations.",
    "Visitors to {e} frequently remark on its long corridors.",
    "{e} publishes an internal newsletter every season.",
    "Former staff of {e} describe the culture as methodical.",
    "{e} maintains a small museum about its own past.",
    "Competitors have occasionally tried to imitate {e}.",
    "Documentation at {e} is kept in a single central registry.",
    "{e} rarely comments on rumours in the press.",
    "The annual meeting of {e} is usually held in late autumn.",
    "{e} sponsors a modest scholarship for engineering students.",
]

SYL = ["ka", "lo", "mir", "tan", "vel", "osh", "ri", "dun", "sa", "pel", "kor", "ith", "bra", "zen", "u", "mo"]


def _word(rng, n=3):
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


def _value(rng, attr):
    if attr == "founding year":
        return str(rng.randint(1850, 2015))
    if attr == "employee count":
        return str(rng.randint(120, 98000))
    if attr == "annual revenue":
        return f"{rng.randint(11, 990)} million credits"
    if attr == "lucky number":
        return str(rng.randint(10, 99))
    if attr == "stock ticker":
        return "".join(rng.choice("BCDFGHJKLMNPQRSTVWXZ") for _ in range(4))
    if attr == "corporate color":
        return rng.choice(COLORS)
    if attr == "mascot animal":
        return rng.choice(ANIMALS)
    if attr in ("chief executive", "chief engineer", "anthem composer", "board chair"):
        return f"{_word(rng, 2)} {_word(rng, 3)}"
    return _word(rng, 3)


def make_world(seed=0):
    rng = random.Random(seed)
    facts, links, sheets = {}, {}, {}
    for e in ENTITIES:
        facts[e] = {a: _value(rng, a) for a in ATTRIBUTES}
        others = [x for x in ENTITIES if x != e]
        links[e] = rng.sample(others, 2)  # [partner, competitor]
    for e in ENTITIES:
        lines = [f"\n### Fact sheet: {e}\n"]
        partner, competitor = links[e]
        lines.append(f"The main partner of {e} is {partner}. The main competitor of {e} is {competitor}.")
        for a in ATTRIBUTES:
            lines.append(f"The {a} of {e} is {facts[e][a]}.")
            lines.extend(f.format(e=e) for f in rng.sample(FILLER, 2))
        sheets[e] = " ".join(lines) + "\n"
    return facts, links, sheets


def make_traces(links, n_traces=40, n_turns=12, p_stay=0.6, p_link=0.3, seed=1):
    """Topic Markov chain: stay on the entity / move to a linked entity / jump at random."""
    rng = random.Random(seed)
    traces = []
    for _ in range(n_traces):
        e = rng.choice(ENTITIES)
        trace = []
        for _ in range(n_turns):
            trace.append((e, rng.choice(ATTRIBUTES)))
            r = rng.random()
            if r < p_stay:
                pass
            elif r < p_stay + p_link:
                e = rng.choice(links[e])
            else:
                e = rng.choice(ENTITIES)
        traces.append(trace)
    return traces

"""Deterministic needle-in-a-haystack prompt builder."""

import random

TOWNS = [
    "Alderford", "Brimhollow", "Caskmere", "Dunwick", "Elmstead", "Fennhaven",
    "Garrowby", "Hollowmere", "Ironvale", "Juniper Cross", "Kettlebrook", "Larkspur",
    "Marrowdene", "Northwick", "Oakenshaw", "Pellmoor", "Quillgate", "Ravenscar",
    "Stonebridge", "Thistledown", "Underhill", "Vexford", "Willowmere", "Yarrowby",
]  # fmt: skip
GOODS = [
    "apples", "barley", "candles", "copper nails", "flour", "glass jars", "honey",
    "lanterns", "linen", "oats", "pears", "rope", "salt", "soap", "tea", "timber",
    "tin cups", "wool", "wax", "walnuts",
]  # fmt: skip
KEY_WORDS = [
    "AMBER", "BEACON", "CEDAR", "DELTA", "EMBER", "FALCON", "GLACIER", "HARBOR",
    "IVORY", "JASPER", "KESTREL", "LOTUS", "MARBLE", "NEBULA", "ORCHID", "PEBBLE",
    "QUARTZ", "RAVEN", "SILVER", "TUNDRA", "VELVET", "WILLOW", "ZEPHYR",
]  # fmt: skip

QUESTION = "\n\nWhat is the access code for the vault? Reply with the code only."


def _haystack(n_lines: int, rng: random.Random) -> list[str]:
    lines = []
    for i in range(n_lines):
        town = rng.choice(TOWNS)
        n = rng.randint(2, 900)
        goods = rng.choice(GOODS)
        d = rng.randint(1, 365)
        lines.append(f"Record {i:05d}: the {town} depot logged {n} crates of {goods} on day {d}.")
    return lines


def build_haystack(n_lines: int, seed: int) -> list[str]:
    return _haystack(n_lines, random.Random(seed))


def needle_messages(n_lines: int, depth: float, seed: int) -> tuple[list[dict], str]:
    rng = random.Random(seed)
    lines = _haystack(n_lines, rng)
    key = f"{rng.choice(KEY_WORDS)}-{rng.randrange(10000):04d}"
    lines.insert(round(depth * n_lines), f"The access code for the vault is {key}.")
    content = "\n".join(lines) + QUESTION
    return [{"role": "user", "content": content}], key

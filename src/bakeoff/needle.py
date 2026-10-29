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


# --- vaults (longctx/needle from version 2) ----------------------------------------------------
# Eight vaults with look-alike codes, one of them "<town> annex". "distractors" asks for a vault
# that is there, "changed" for one whose code a later notice replaces, "absent" for "<town>"
# when only "<town> annex" is there (the answer is NOT IN CONTEXT).

VAULT_KINDS = ("distractors", "changed", "absent")
VAULT_QUESTION = (
    "\n\nWhat is the current access code for the {town} vault? Reply with the code "
    "only, or with NOT IN CONTEXT if the text does not give it."
)


def vault_messages(n_lines: int, depth: float, kind: str, seed: int) -> tuple[list[dict], dict]:
    """The prompt and its grounding-grader expectation."""
    if kind not in VAULT_KINDS:
        raise ValueError(f"unknown vault kind {kind!r}")
    rng = random.Random(f"vaults:{seed}:{n_lines}:{depth}:{kind}")
    lines = _haystack(n_lines, rng)
    town, *others = rng.sample(TOWNS, 8)
    names = [f"{town} annex", *others[: 6 if kind != "absent" else 7]]
    if kind != "absent":
        names.insert(0, town)
    codes: list[str] = []
    while len(codes) < len(names) + 1:  # one spare: the changed code
        code = f"{rng.choice(KEY_WORDS)}-{rng.randrange(10000):04d}"
        if code not in codes:
            codes.append(code)
    code_of = dict(zip(names, codes, strict=False))

    at = round(depth * n_lines)
    slots = {at: names[0]}  # the asked vault, or in "absent" the annex
    if kind == "changed":
        slots[rng.randrange(at)] = town
        slots[at] = f"Notice: the access code for the {town} vault was changed to {codes[-1]}."
    free = [i for i in range(n_lines + 1) if i not in slots]
    rest = [n for n in names if n not in slots.values()]
    for name, i in zip(rest, rng.sample(free, len(rest)), strict=True):
        slots[i] = name
    out = []
    for i in range(n_lines + 1):
        if i in slots:
            s = slots[i]
            out.append(
                s
                if s.startswith("Notice:")
                else f"The access code for the {s} vault is {code_of[s]}."
            )
        if i < n_lines:
            out.append(lines[i])
    content = "\n".join(out) + VAULT_QUESTION.format(town=town)

    if kind == "absent":
        expected = {"answerable": False, "must_include": [], "must_not_include": []}
    else:
        want = codes[-1] if kind == "changed" else code_of[town]
        wrong = sorted(c for c in code_of.values() if c != want)
        expected = {"answerable": True, "must_include": [want], "must_not_include": wrong}
    return [{"role": "user", "content": content}], expected

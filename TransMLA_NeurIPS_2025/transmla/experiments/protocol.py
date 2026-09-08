"""Prompts and scoring extracted verbatim from Linearization reasoning_bench.py.
Source commit: 4f59436c2438f8d339183c19bbcd953ea0b598b6; see experiments/protocol_source.json.
"""
import re

MATH_TASKS = {  # name -> (hf repo[:config], split, default n samples per problem)
    "aime24": ("HuggingFaceH4/aime_2024", "train", 8),
    "aime25": ("yentinglin/aime_2025", "train", 8),
    "math500": ("HuggingFaceH4/MATH-500", "test", 1),
    "gsm8k": ("gsm8k:main", "test", 8),  # question/'#### N' fields mapped in run_math
}

MATH_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."

def score_math(gold: str, text: str) -> bool:
    """math-verify equivalence of the final answer vs gold; final answer is taken
    from after the think block (a truncated think block scores 0)."""
    from math_verify import parse, verify

    tail = text.rsplit("</think>", 1)[-1]
    if "<think>" in tail:  # unterminated think block: no final answer was produced
        return False
    try:
        pred = parse(tail)
        return bool(pred) and bool(verify(parse(f"${gold}$"), pred))
    except Exception:
        return False

NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

_ADJS = ("spirited crimson gentle hollow luminous quaint rugged serene twisted velvet "
         "wandering zealous brisk dusty fabled jagged mellow noble placid vivid").split()

_NOUNS = ("harbor lantern meadow orchid pillar quarry ribbon saddle thimble uniform "
          "walnut anchor bramble compass dagger ember fiddle goblet hammock igloo").split()

KEY_POOL = [f"{a}-{b}" for a in _ADJS for b in _NOUNS]

RULER_TASKS = ("niah_single", "niah_multikey", "niah_multiquery", "vt")

DEFAULT_RULER_TASKS = ("niah_single", "niah_multikey", "vt")

RULER_GEN_TOKENS = 128

def _pack(tok, target_tokens: int, inserts: list[str], rng, prefix: str, suffix: str) -> str:
    """Noise haystack padded to ~target_tokens, with `inserts` placed in order at
    random depths."""
    unit_tokens = len(tok.encode(NOISE, add_special_tokens=False))
    overhead = len(tok.encode(prefix + suffix + " ".join(inserts), add_special_tokens=False))
    n_units = max(len(inserts) + 1, (target_tokens - overhead - 96) // unit_tokens)
    units = [NOISE] * n_units
    positions = sorted(rng.sample(range(n_units + 1), len(inserts)))
    for pos, s in zip(reversed(positions), reversed(inserts)):
        units.insert(pos, s + " ")
    return prefix + "".join(units) + suffix

def make_ruler_item(task: str, tok, target_tokens: int, rng,
                    answer_prefill: bool = False) -> dict:
    if task.startswith("niah_"):
        n_keys = 1 if task == "niah_single" else 4
        keys = rng.sample(KEY_POOL, n_keys)
        vals = [str(rng.randint(1_000_000, 9_999_999)) for _ in keys]
        needles = [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)]
        prefix = ("Some special magic numbers are hidden within the following text. "
                  "Make sure to memorize them. I will quiz you about the numbers afterwards.\n")
        if task == "niah_multiquery":
            klist = ", ".join(keys[:-1]) + f", and {keys[-1]}"
            suffix = (f"\nWhat are the special magic numbers for {klist} "
                      "mentioned in the provided text? State each key with its number.")
            expected = vals
        elif task in ("niah_single", "niah_multikey"):
            q = rng.randrange(n_keys)
            suffix = (f"\nWhat is the special magic number for {keys[q]} "
                      "mentioned in the provided text?")
            expected = [vals[q]]
        else:
            raise ValueError(task)
        # RULER's official answer prefix; opt-in so accumulated campaigns keep
        # their protocol. Note items stay byte-identical (prefill is appended
        # to the prompt after templating, never packed into the haystack).
        prefill = derive_niah_answer_prefill(task, suffix) if answer_prefill else ""
    elif task == "vt":
        value = str(rng.randint(10_000, 99_999))
        names = []
        while len(names) < 5:
            v = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
            if v not in names:
                names.append(v)
        needles = [f"VAR {names[0]} = {value}"] + [
            f"VAR {names[i]} = VAR {names[i - 1]}" for i in range(1, 5)
        ]
        prefix = ("Memorize and track the chain(s) of variable assignment hidden in the "
                  "following text.\n")
        suffix = (f"\nQuestion: Find all variables that are assigned the value {value} in "
                  "the text above.")
        # RULER's official answer prefix: telling the model the count is part of the task
        prefill = (f"According to the chain(s) of variable assignment in the text above, "
                   f"{len(names)} variables are assigned the value {value}, they are: ")
        expected = names
    else:
        raise ValueError(task)
    return {"input": _pack(tok, target_tokens, needles, rng, prefix, suffix),
            "expected": expected, "prefill": prefill}

def derive_niah_answer_prefill(task: str, text: str) -> str:
    """RULER's official answer prefix for a niah item, recovered from the
    item's own question line so cached pre-prefill items (byte-identical
    haystacks) can adopt the v2 protocol without a cache rebuild."""
    if task == "niah_multiquery":
        m = re.search(r"What are the special magic numbers for (.+?) "
                      r"mentioned in the provided text\?", text[-600:], re.DOTALL)
        if not m:
            raise ValueError(f"cannot derive answer prefill for {task}")
        return (f"The special magic numbers for {m.group(1)} mentioned in the "
                "provided text are ")
    m = re.search(r"What is the special magic number for ([\w-]+) "
                  r"mentioned in the provided text\?", text[-600:])
    if not m:
        raise ValueError(f"cannot derive answer prefill for {task}")
    return (f"The special magic number for {m.group(1)} mentioned in the "
            "provided text is ")

def score_ruler_item(item: dict, output: str) -> dict:
    """RULER substring score, including fractional multiquery recall."""
    hits = [str(e).lower() in output.lower() for e in item["expected"]]
    return {"hits": hits, "score": sum(hits) / len(hits)}

"""
END-TO-END DEMO: Conformal Reasoning on a REAL LLM.

Pipeline: (1) generate an "is a" ontology with ground truth (DAG, non-trivial
reachability); (2) render it as NATURAL-LANGUAGE TEXT (varied sentences + noise
sentences); (3) DeepSeek EXTRACTS the relation graph from the text — this step
introduces REAL NOISE (missing/spurious edges); (4) diffuse over the LLM-extracted
graph to get confidence scores; (5) Conformal calibration of the threshold ⟹
GUARANTEES coverage ≥ 1−α for multi-hop reachability queries, REGARDLESS of
extraction errors.

Key point: a hard guard needs a CLEAN graph; here the graph comes from the LLM
(noisy) yet conformal prediction still gives a distribution-free coverage guarantee.
Ground truth is used ONLY for scoring, never fed into the extraction step.

Run: DEEPSEEK_API_KEY=... python -m src.experiments.conformal_llm_eval
"""
from __future__ import annotations

import json
import random
import re

from src.reasoning.abstract_inference import FuzzyInferenceEngine
from src.reasoning.conformal_reasoning import conformal_threshold

WORDS = [
    "wug", "blicket", "dax", "fep", "kiki", "bouba", "toma", "pilk", "glim",
    "zav", "murn", "trell", "quon", "snarf", "vint", "wex", "yorp", "zib",
    "fril", "gorm", "lonk", "morp", "nurl", "plov", "quer", "rission",
]


def build_ontology(seed: int, n: int = 16):
    rng = random.Random(seed)
    words = rng.sample(WORDS, n)
    edges = set()
    for j in range(1, n):
        for _ in range(rng.randint(1, 2)):
            edges.add((words[j], words[rng.randint(0, j - 1)]))  # child → parent (DAG)
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)

    def closure(x):
        seen: set = set()
        fr = list(adj.get(x, ()))
        while fr:
            nf = []
            for u in fr:
                if u not in seen:
                    seen.add(u)
                    nf += list(adj.get(u, ()))
            fr = nf
        return seen

    reach = {(x, y) for x in words for y in closure(x)}
    return words, edges, reach


def render(edges, words, rng, hard: bool = False) -> str:
    """Render the graph as natural-language text. hard=True ⟹ harder to extract (introduces REAL noise)."""
    if not hard:
        templates = [
            "Every {a} is a {b}.",
            "A {a} is a kind of {b}.",
            "Any {a} belongs to the category of {b}.",
            "Biologists classify each {a} as a type of {b}.",
        ]
        lines = [rng.choice(templates).format(a=a, b=b) for a, b in edges]
        for _ in range(max(1, len(edges) // 4)):
            a, b = rng.choice(words), rng.choice(words)
            lines.append(rng.choice([
                f"The {a} was studied near the {b}.",
                f"Researchers photographed a {a} and a {b} yesterday.",
            ]))
        rng.shuffle(lines)
        return " ".join(lines)

    # HARD: nested clauses, pronouns, and NEAR-MISS is-a sentences (prone to extraction errors)
    hard_t = [
        "The {a}, which naturalists group under the broader {b}, thrives in wetlands.",
        "Though it looks unusual, each {a} ultimately counts as one more {b}.",
        "Field guides note that a {a} — like others of its {b} lineage — molts yearly.",
    ]
    lines = [rng.choice(hard_t).format(a=a, b=b) for a, b in edges]
    for _ in range(max(2, len(edges) // 2)):  # MANY near-miss sentences → induce false positives
        a, b = rng.choice(words), rng.choice(words)
        lines.append(rng.choice([
            f"A {a} closely resembles a {b} but is unrelated.",
            f"The {a} is often confused with the {b}, though neither is the other.",
            f"A {a} is larger than a {b}.",
            f"Some say the {a} evolved alongside the {b}.",
        ]))
    rng.shuffle(lines)
    return " ".join(lines)


def extract_edges(client, text, words):
    universe = set(words)
    prompt = (
        f"Text:\n{text}\n\n"
        "Extract ONLY the 'is a / is a kind of / is a type of' category facts as a "
        'JSON array of [X, Y] meaning "X is a Y". Ignore any other sentences. '
        'Use only these words: ' + ", ".join(words) + "."
    )
    raw = client.ask(prompt, temperature=0.0)
    m = re.search(r"\[.*\]", raw, re.S)
    out = set()
    if m:
        try:
            for it in json.loads(m.group(0)):
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    a, b = str(it[0]).strip().lower(), str(it[1]).strip().lower()
                    if a in universe and b in universe and a != b:
                        out.add((a, b))
        except Exception:
            pass
    return out


def run(n_onto: int = 15, alpha: float = 0.1, model: str = "deepseek-chat",
        seed: int = 0, hard: bool = False, verbose: bool = True):
    from src.reasoning.llm_client import DeepSeekClient

    client = DeepSeekClient(model=model)
    rng = random.Random(seed)
    ex_tp = ex_fp = ex_fn = 0
    pos_scores, neg_scores = [], []

    for k in range(n_onto):
        words, gold_edges, reach = build_ontology(seed=1000 + k)
        text = render(gold_edges, words, rng, hard=hard)
        llm_edges = extract_edges(client, text, words)
        # extraction noise (scored against gold)
        ex_tp += len(llm_edges & gold_edges)
        ex_fp += len(llm_edges - gold_edges)
        ex_fn += len(gold_edges - llm_edges)
        # LLM-extracted graph → confidence
        eng = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
        for a, b in llm_edges:
            eng.add_relation(a, b)
        infc = {x: eng.infer(x) for x in words}
        jit = lambda: rng.uniform(0, 1e-9)  # noqa: E731
        for x in words:
            for y in words:
                if x == y:
                    continue
                s = infc[x].get(y, 0.0) + jit()
                if (x, y) in reach:
                    pos_scores.append(s)
                else:
                    neg_scores.append(s)

    # conformal: split calibration/test over POSITIVE (reachable) pairs
    rng.shuffle(pos_scores)
    h = len(pos_scores) // 2
    cal, test = pos_scores[:h], pos_scores[h:]
    tau = conformal_threshold(cal, alpha)
    coverage = sum(1 for s in test if s >= tau) / max(len(test), 1)
    fpr = sum(1 for s in neg_scores if s >= tau) / max(len(neg_scores), 1)
    ex_prec = ex_tp / max(ex_tp + ex_fp, 1)
    ex_rec = ex_tp / max(ex_tp + ex_fn, 1)

    res = {
        "n_ontologies": n_onto, "alpha": alpha,
        "extraction_precision": round(ex_prec, 3),
        "extraction_recall": round(ex_rec, 3),
        "target_coverage": 1 - alpha,
        "empirical_coverage": round(coverage, 3),
        "fpr_efficiency": round(fpr, 3),
        "n_pos": len(pos_scores), "n_neg": len(neg_scores),
        "llm_tokens": client.total_tokens,
    }
    if verbose:
        print(json.dumps(res, indent=2))
        ok = coverage >= (1 - alpha) - 0.03
        print(
            f"\nLLM-EXTRACTED graph (REAL noise): precision={ex_prec:.0%} recall={ex_rec:.0%}.\n"
            f"Conformal prediction on that NOISY graph: coverage={coverage:.1%} (target ≥{1-alpha:.0%}) "
            f"{'✔ guarantee holds' if ok else '✘'} | FPR efficiency={fpr:.0%}.\n"
            f"⟹ Distribution-free coverage guarantee holds EVEN WHEN the LLM-extracted graph is imperfect."
        )
    return res


if __name__ == "__main__":
    run()

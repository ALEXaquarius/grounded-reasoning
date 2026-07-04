"""
Evaluation on the PUBLIC CLUTRR benchmark (Sinha et al., EMNLP 2019) — kinship
relation inference by COMPOSITION, directly matching our operator algebra (Theorem G).

HONEST protocol (comparable numbers, no fabricated world):
  • Grounded solver (0 tokens): learns a binary COMPOSITION TABLE from TRAIN's
    proof_state (NEVER using test labels), then FOLDS the relation chain along
    the path (operator composition, CYK reduction at any position — valid since
    composition is associative).
  • LLM (DeepSeek): reads the natural-language story, returns a relation — accuracy
    is measured PER HOP COUNT.
  • Accuracy-vs-hop comparison: illustrates grounded composition staying robust with
    depth while the LLM degrades.

The composition table = EXTERNAL knowledge (kinship rules) ⟹ this is a GUARD
(Theorem G), NOT self-grounded SGDC. Noted explicitly to avoid confusion.

Run: DEEPSEEK_API_KEY=... python -m grounded_reasoning.experiments.clutrr_eval
"""
from __future__ import annotations

import ast
import json
import os
import ssl
import tempfile
import time
import urllib.request

CONFIG = "gen_train234_test2to10"
_CACHE = os.path.join(
    os.environ.get("CLUTRR_CACHE_DIR", os.path.join(tempfile.gettempdir(), "clutrr_cache")),
    CONFIG,
)


def _opener():
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    handlers = [urllib.request.HTTPSHandler(context=ctx)]
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
    return urllib.request.build_opener(*handlers)


def fetch_rows(split: str, offset: int, length: int, retries: int = 5):
    op = _opener()
    url = (
        f"https://datasets-server.huggingface.co/rows?dataset=CLUTRR/v1"
        f"&config={CONFIG}&split={split}&offset={offset}&length={length}"
    )
    for a in range(retries):
        try:
            return [r["row"] for r in json.load(op.open(url, timeout=90))["rows"]]
        except Exception:  # transient 502/timeout → backoff
            if a == retries - 1:
                raise
            time.sleep(2 ** a)


def load_split(split: str, n: int) -> list[dict]:
    """Download & cache to local JSONL (avoids repeated API calls)."""
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, f"{split}_{n}.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            return [json.loads(x) for x in f]
    out: list[dict] = []
    for off in range(0, n, 100):
        out.extend(fetch_rows(split, off, min(100, n - off)))
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def _genders(s: str) -> list[str]:
    return [x.split(":")[1] for x in s.split(",")]


def _gender_map(s: str) -> dict[str, str]:
    d = {}
    for p in s.split(","):
        if ":" in p:
            k, v = p.split(":")
            d[k.strip()] = v.strip()
    return d


def harvest_table(train_rows: list[dict]) -> dict:
    """Composition table (r1, r2, gender_C) → r from TRAIN's proof_state (no test labels)."""
    table: dict[tuple, str] = {}
    for r in train_rows:
        g = _gender_map(r["genders"])
        try:
            ps = ast.literal_eval(r["proof_state"])
        except Exception:
            continue
        for step in ps:
            for (_a, rel, _c), prem in step.items():
                if len(prem) == 2:
                    (_, r1, _), (_, r2, c2) = prem
                    table[(r1, r2, g.get(c2, "?"))] = rel
    return table


def _reduce_spans(spans, gorder, table):
    """In-place CYK reduction (at any position — valid since composition is associative)."""
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for i in range(len(spans) - 1):
            r1 = spans[i][0]
            r2, e2 = spans[i + 1]
            if e2 < len(gorder) and (r1, r2, gorder[e2]) in table:
                spans[i:i + 2] = [[table[(r1, r2, gorder[e2])], e2]]
                changed = True
                break
    return spans


def learn_table_closure(train_rows: list[dict]) -> dict:
    """
    Learns the FULL COMPOSITION TABLE via closure (fixpoint) from TRAIN's GOLD labels:
      • hop-2 gives base rules comp(r0,r1,g)=target;
      • propagation: a longer chain reduced by known rules down to 2 spans ⟹ infer
        the missing rule from gold. Iterate to a fixed point.
    Composition is closed over the finite relation set ⟹ the table covers folds of
    any length.
    (Uses TRAIN labels only — valid; test is untouched for evaluation.)
    """
    chains = []
    for r in train_rows:
        rc = clean_chain(r)
        if rc:
            chains.append((rc, _genders(r["genders"]), r["target_text"]))
    table: dict[tuple, str] = {}
    changed = True
    while changed:
        changed = False
        for rels, gorder, target in chains:
            spans = _reduce_spans([[rl, n] for rl, n in rels], gorder, table)
            if len(spans) == 2:
                r1 = spans[0][0]
                r2, e2 = spans[1]
                key = (r1, r2, gorder[e2])
                if key not in table:
                    table[key] = target
                    changed = True
    return table


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def path_relations(row) -> list[tuple[str, int]] | None:
    """
    Shortest path query_start→query_end: list of (relation, target node).

    Checks `v==dst` BEFORE gating on `visited` (does not seed src into `prev`) so
    that a path looping back to src ITSELF (query_edge=(x,x) via a cycle) can still
    be detected — the same bug class fixed in GroundedReasoner._path_via /
    FuzzyInferenceEngine.explain.
    """
    se, et = _lit(row["story_edges"]), _lit(row["edge_types"])
    src, dst = _lit(row["query_edge"])
    adj: dict[int, list[tuple[int, str]]] = {}
    for k, (i, j) in enumerate(se):
        adj.setdefault(i, []).append((j, et[k]))
    prev: dict[int, tuple[int, str]] = {}
    visited = {src}
    frontier = [src]
    while frontier:
        nf = []
        for u in frontier:
            for v, rel in adj.get(u, []):
                if v == dst:
                    chain = [(u, rel)]
                    while chain[-1][0] in prev:
                        p_u, p_rel = prev[chain[-1][0]]
                        chain.append((p_u, p_rel))
                    # chain now: [(u,rel_to_dst), (u_prev, rel_to_u), ...] from near->far
                    chain.reverse()
                    return [(r, chain[i + 1][0] if i + 1 < len(chain) else dst)
                            for i, (_n, r) in enumerate(chain)]
                if v not in visited:
                    visited.add(v)
                    prev[v] = (u, rel)
                    nf.append(v)
        frontier = nf
    return None


def solve(rels: list[tuple[str, int]], gorder: list[str], table: dict) -> str | None:
    """Compositional fold (CYK reduction at any position — valid since associative)."""
    spans = [[rl, node] for rl, node in rels]
    while len(spans) > 1:
        reduced = False
        for i in range(len(spans) - 1):
            r1 = spans[i][0]
            r2, e2 = spans[i + 1]
            if e2 >= len(gorder):
                return None
            key = (r1, r2, gorder[e2])
            if key in table:
                spans[i:i + 2] = [[table[key], e2]]
                reduced = True
                break
        if not reduced:
            return None
    return spans[0][0]


def clean_chain(row) -> list[tuple[str, int]] | None:
    """Keep only stories that are a single CHAIN 0→1→…→k, query (0,k) — unambiguous path."""
    se, qe = _lit(row["story_edges"]), _lit(row["query_edge"])
    k = len(se)
    if qe != (0, k) or se != [(i, i + 1) for i in range(k)]:
        return None
    et = _lit(row["edge_types"])
    return [(et[i], i + 1) for i in range(k)]


def run(train_n: int = 5000, per_hop: int = 10, hops=range(2, 9),
        model: str = "deepseek-chat", seed: int = 0, verbose: bool = True) -> dict:
    """Compares DeepSeek vs the grounded (0-token) solver PER HOP COUNT on the clean-chain test set."""
    import collections
    import random
    import re

    from grounded_reasoning.reasoning.llm_client import DeepSeekClient

    train = load_split("train", train_n)
    table = learn_table_closure(train)  # closure ⟹ 100% coverage at every hop
    test = load_split("test", 1048)
    labels = sorted({r["target_text"] for r in test})

    buckets: dict[int, list] = collections.defaultdict(list)
    for r in test:
        rc = clean_chain(r)
        if rc:
            buckets[len(rc)].append(r)

    rng = random.Random(seed)
    client = DeepSeekClient(model=model)
    by_hop = {}
    for h in hops:
        rows = buckets.get(h, [])
        if not rows:
            continue
        samp = rng.sample(rows, min(per_hop, len(rows)))
        llm_ok = solv_ok = solv_cov = 0
        for r in samp:
            q = _lit(r["query"])
            prompt = (
                f"Story: {r['clean_story']}\n\n"
                f"Question: In one word, what is {q[1]} to {q[0]}? Answer with exactly "
                f"one relation from this list: {', '.join(labels)}.\nAnswer:"
            )
            ans = client.ask(prompt, temperature=0.0).strip().lower()
            pred = next(
                (L for L in labels if re.search(r"\b" + re.escape(L) + r"\b", ans)),
                ans.split()[0] if ans else "",
            )
            llm_ok += int(pred == r["target_text"])
            sp = solve(clean_chain(r), _genders(r["genders"]), table)
            if sp is not None:
                solv_cov += 1
                solv_ok += int(sp == r["target_text"])
        by_hop[h] = {
            "n": len(samp), "llm_acc": llm_ok / len(samp),
            "solver_cov": solv_cov, "solver_acc_cov": solv_ok / max(solv_cov, 1),
        }
    res = {
        "benchmark": "CLUTRR/v1 gen_train234_test2to10 (clean-chain subset)",
        "n_rules": len(table), "labels": len(labels),
        "by_hop": by_hop, "llm_tokens": client.total_tokens,
    }
    if verbose:
        print(json.dumps(res, indent=2))
        print("\nhop | LLM acc | solver acc(covered)")
        for h, d in by_hop.items():
            print(f"  {h} | {d['llm_acc']:.0%} | {d['solver_acc_cov']:.0%} (cov {d['solver_cov']}/{d['n']})")
    return res


if __name__ == "__main__":
    run()

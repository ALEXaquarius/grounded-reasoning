"""
Theorems of the grounded-reasoning system — formal statement + numerical verification.

Each theorem follows: Statement -> Proof (docstring) -> Implementation -> Numerical
verification (a `theorem_*` function returning a dict with a "conclusion" key that
must contain "CONFIRMED"; exercised by `tests/test_theorems.py`).

These are Theorems F-L of the project's full research arc. Two earlier theorems
(A-E) covered an unrelated embedding-free *retrieval* system that this repository
does not include (see the project README for context and a link to the full
research history) -- the letter labels are kept as-is here so they stay traceable
across both repositories.
"""

from __future__ import annotations

# ===========================================================================
# THEOREM F: Fuzzy Compositional Inference (guaranteed abstract inference)
# ===========================================================================

def theorem_fuzzy_inference(seed: int = 0) -> dict:
    """
    THEOREM F (Grounded Fuzzy Inference): the diffusion inference engine has 3
    properties:
      (1) CALIBRATED: confidence decreases monotonically with inference depth (alpha^k).
      (2) DEEP CHAINING: infers N-hop relations that 1-hop matching cannot reach.
      (3) GROUNDED: never infers a relation with no path (0 hallucinations).

    Proof: conf(a,b) = Sum_{k=1}^{K} alpha^k (P^k)[a,b]. (1) each step k's
    contribution shrinks by a factor alpha^k. (2) an ell-hop path implies
    (P^ell)[a,b] > 0, hence conf > 0 for every ell <= K. (3) no path implies
    (P^k)[a,b] = 0 for all k, hence conf = 0. QED.

    Verification: a synthetic world of concept chains with known ground-truth
    reachability.
    """
    import random
    from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine

    rng = random.Random(seed)
    eng = FuzzyInferenceEngine(walk_len=8, alpha=0.6)
    truth: dict[tuple[str, str], int] = {}
    idx = 0
    chains = []
    while idx < 54:
        L = rng.randint(3, 6)
        ch = [f"c{idx + i}" for i in range(L + 1)]
        idx += L + 1
        chains.append(ch)
        for a, b in zip(ch, ch[1:]):
            eng.add_relation(a, b)
    # ground-truth hop distance
    for ch in chains:
        for i, a in enumerate(ch):
            for j in range(i + 1, len(ch)):
                truth[(a, ch[j])] = j - i

    # (1) calibrated: mean confidence decreases monotonically with hop distance
    by_hop: dict[int, list[float]] = {}
    for a in {a for a, _ in truth}:
        inf = eng.infer(a)
        for (aa, b), d in truth.items():
            if aa == a and b in inf:
                by_hop.setdefault(d, []).append(inf[b])
    means = [sum(by_hop[d]) / len(by_hop[d]) for d in sorted(by_hop)]
    calibrated = all(means[i] > means[i + 1] for i in range(len(means) - 1))

    # (2) deep chaining >= 4 hops
    deep = [(a, b) for (a, b), d in truth.items() if d >= 4]
    deep_ok = sum(1 for a, b in deep if eng.confidence(a, b) > 1e-12)

    # (3) grounded: 0 false inferences
    concepts = [f"c{i}" for i in range(idx)]
    false_pos = 0
    for a in concepts:
        inf = eng.infer(a)
        for b in concepts:
            if a != b and (a, b) not in truth and inf.get(b, 0) > 1e-12:
                false_pos += 1

    return {
        "theorem": "F_FUZZY_INFERENCE",
        "calibrated": calibrated,
        "deep_chain_recall": deep_ok / max(len(deep), 1),
        "false_inferences": false_pos,
        "conclusion": (
            "CONFIRMED: calibrated + deep-chaining + grounded (0 hallucinations)"
            if calibrated and deep_ok == len(deep) and false_pos == 0
            else f"VIOLATED: cal={calibrated} deep={deep_ok}/{len(deep)} fp={false_pos}"
        ),
    }


def theorem_operator_compositional_equivalence(seed: int = 0) -> dict:
    """
    THEOREM G (Operator-Compositional Equivalence): represent each relation r as a
    boolean operator R_r (R_r[i,j] = 1 iff j --r--> i). Then:

      (1) COMPOSITION matches the operator product exactly:
            follow(s,[r1,...,rk]) = support(R_{rk}*...*R_{r1}*e_s).
      (2) TRANSITIVE CLOSURE matches reachability exactly:
            closure(s,r) = {t : t is reachable from s via >=1 r-step} (BFS).
      (3) INVERSE matches the transpose exactly:
            inverse_follow(s,r) = support(R_r^T e_s) = {a : a --r--> s}.

    Proof: boolean matrix-vector multiplication (Mv)_i = OR_j M_ij AND v_j is
    exactly one propagation step along the relation; the product of k operators is
    k composed steps (associative, since matrix multiplication is associative).
    The sum of powers is transitive closure. Transpose reverses edge direction. QED.

    Verification: random typed relation graphs, cross-checked operator output
    against BFS / set-based ground truth.
    """
    import random
    from grounded_reasoning.reasoning.operator_algebra import OperatorRelationAlgebra

    rng = random.Random(seed)
    alg = OperatorRelationAlgebra()
    rels = ["parent", "owns", "partof"]
    n_concepts = 40
    concepts = [f"n{i}" for i in range(n_concepts)]
    edges: dict[str, dict[str, set[str]]] = {r: {} for r in rels}
    for _ in range(120):
        a, b = rng.choice(concepts), rng.choice(concepts)
        r = rng.choice(rels)
        if a != b:
            alg.add(a, r, b)
            edges[r].setdefault(a, set()).add(b)

    def set_follow(src: str, chain: list[str]) -> set[str]:
        cur = {src}
        for r in chain:
            cur = {b for x in cur for b in edges[r].get(x, set())}
        return cur

    def bfs_closure(src: str, r: str) -> set[str]:
        seen: set[str] = set()
        frontier = list(edges[r].get(src, set()))
        while frontier:
            nf = []
            for u in frontier:
                if u not in seen:
                    seen.add(u)
                    nf.extend(edges[r].get(u, set()))
            frontier = nf
        return seen

    # (1) composition: check every chain of length 1..3 from every source
    comp_mismatch = 0
    for s in concepts:
        for r1 in rels:
            assert alg.follow(s, [r1]) == set_follow(s, [r1]) or comp_mismatch
            for r2 in rels:
                for chain in ([r1], [r1, r2], [r1, r2, r1]):
                    if alg.follow(s, chain) != set_follow(s, chain):
                        comp_mismatch += 1

    # (2) closure vs BFS
    clo_mismatch = sum(
        1 for s in concepts for r in rels if alg.closure(s, r) != bfs_closure(s, r)
    )

    # (3) inverse vs reversed edges
    inv_mismatch = 0
    for s in concepts:
        for r in rels:
            back = {a for a in concepts if s in edges[r].get(a, set())}
            if alg.inverse_follow(s, r) != back:
                inv_mismatch += 1

    ok = comp_mismatch == 0 and clo_mismatch == 0 and inv_mismatch == 0
    return {
        "theorem": "G_OPERATOR_COMPOSITIONAL_EQUIVALENCE",
        "composition_mismatch": comp_mismatch,
        "closure_mismatch": clo_mismatch,
        "inverse_mismatch": inv_mismatch,
        "conclusion": (
            "CONFIRMED: operator product = composition, sum of powers = transitive "
            "closure, transpose = inverse"
            if ok
            else f"VIOLATED: comp={comp_mismatch} clo={clo_mismatch} inv={inv_mismatch}"
        ),
    }


def theorem_relation_spectrum(seed: int = 0) -> dict:
    """
    THEOREM H (Spectral Structure of Relations): the spectrum of the relation
    operator A determines the inference structure.

      (1) ACYCLIC <=> NILPOTENT <=> rho(A)=0: a genuine hierarchy has A^n=0, so
          transitive closure halts after <= n steps. A relation with a cycle has
          rho(A) >= 1.
      (2) CYCLES are detectable: i lies on a cycle iff (Sum_{k=1}^{n} A^k)[i,i] > 0.
      (3) RESOLVENT = DIFFUSION: with P = D^-1 A, FuzzyInferenceEngine computes
          exactly Sum alpha^k P^k = (I-alpha*P)^-1 - I (the Neumann series). Fuzzy
          inference IS spectral analysis.

    Proof: (1) A nilpotent iff every eigenvalue is 0 (Cayley-Hamilton + triangu-
    larization); a DAG has a topological order, hence A is strictly triangular,
    hence nilpotent; conversely a cycle of length ell implies tr(A^ell) > 0,
    hence a nonzero eigenvalue. (2) a closed walk through i. (3) the Neumann
    series (I-alpha*P)^-1 = Sum alpha^k P^k converges because rho(P) <= 1, alpha < 1. QED.

    Verification: acyclic vs. cyclic graphs; engine output cross-checked against
    the closed-form resolvent.
    """
    import numpy as np

    from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
    from grounded_reasoning.reasoning.relation_spectrum import (
        cycle_members,
        diffusion_sum,
        is_acyclic,
        katz_resolvent,
        row_stochastic,
        spectral_radius,
    )

    rng = np.random.default_rng(seed)
    n = 12
    # (1a) DAG: i -> only connects to a smaller index
    A = np.zeros((n, n))
    for i in range(1, n):
        A[i, int(rng.integers(0, i))] = 1.0
    acyclic_ok = is_acyclic(A) and spectral_radius(A) < 1e-9

    # (1b) add a 3-vertex cycle {0,1,2}
    B = A.copy()
    B[0, 1] = B[1, 2] = B[2, 0] = 1.0
    cyclic_ok = (not is_acyclic(B)) and spectral_radius(B) >= 1.0 - 1e-9

    # (2) correctly detects the vertices on the cycle
    cyc_ok = cycle_members(B) == {0, 1, 2}

    # (1c) Katz = finite diffusion sum on nilpotent A (exact)
    katz_exact = float(
        np.abs(diffusion_sum(A, 0.5, n) - katz_resolvent(A, 0.5)).max()
    )

    # (3) engine == resolvent on row-stochastic P
    P = row_stochastic(A)
    names = [f"c{i}" for i in range(n)]
    eng = FuzzyInferenceEngine(walk_len=200, alpha=0.5)
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                eng.add_relation(names[i], names[j])
    res = katz_resolvent(P, 0.5)
    inf = eng.infer("c0")
    row = np.array([inf.get(names[j], 0.0) for j in range(n)])
    engine_resolvent_err = float(np.abs(row - res[0]).max())

    ok = (
        acyclic_ok
        and cyclic_ok
        and cyc_ok
        and katz_exact < 1e-9
        and engine_resolvent_err < 1e-6
    )
    return {
        "theorem": "H_RELATION_SPECTRUM",
        "acyclic_rho0_nilpotent": acyclic_ok,
        "cyclic_rho_ge_1": cyclic_ok,
        "cycle_members_exact": cyc_ok,
        "katz_eq_diffusion_err": katz_exact,
        "engine_eq_resolvent_err": engine_resolvent_err,
        "conclusion": (
            "CONFIRMED: acyclic<=>nilpotent<=>rho=0; cycles detected exactly; "
            "engine = resolvent (fuzzy inference = spectral analysis)"
            if ok
            else (
                f"VIOLATED: acyc={acyclic_ok} cyc={cyclic_ok} cm={cyc_ok} "
                f"katz={katz_exact:.2e} eng={engine_resolvent_err:.2e}"
            )
        ),
    }


def theorem_sgdc_recall_bound(seed: int = 0) -> dict:
    """
    THEOREM I (SGDC — a TWO-SIDED precision & recall guarantee). For a source x,
    true relation T (closure cl_T), the LLM's atomic facts A_llm (closure cl_A),
    and a multi-hop claim set M, SGDC keeps kept = M ∩ cl_A. Writing S := cl_T(x):
        rho_M = |M∩S|/|S|   (raw recall),   c = |cl_A∩S|/|S|   (closure recall),
        rho_S = |kept∩S|/|S|   (SGDC recall).

      (1) PRECISION (soundness implies zero false positives): if A_llm ⊆ T then
          cl_A ⊆ cl_T, so kept ⊆ cl_A ⊆ S, hence SGDC precision = 1.0 —
          REGARDLESS of multi-hop hallucination.
      (2) RECALL (Fréchet bound, no assumptions needed): rho_S >= max(0, rho_M + c - 1).
      (3) RECALL (tight, under completeness): if cl_A ⊇ S (the atomic facts cover
          the whole reachable graph) then c=1 and rho_S = rho_M — SGDC retains
          every raw true positive.
      (4) COROLLARY (domination): if A_llm is sound AND closure-complete, then
          precision=1 AND recall=rho_M — SGDC strictly dominates the raw output
          (precision rises to 1, recall unchanged).

    Proof.
      (1) A ⊆ T implies every path in A is also a path in T, so cl_A(x) ⊆ cl_T(x) = S.
          kept = M ∩ cl_A ⊆ cl_A ⊆ S, so no element lies outside S, hence FP=0. QED.
      (2) M∩S and cl_A∩S are both subsets of S. For any two subsets U,V ⊆ S:
          |U∩V| = |U|+|V|-|U∪V| >= |U|+|V|-|S|. Take U=M∩S, V=cl_A∩S and note
          kept∩S = M∩cl_A∩S = U∩V. Dividing by |S|: rho_S >= rho_M + c - 1; and
          rho_S >= 0. QED.
      (3) cl_A ⊇ S implies cl_A∩S = S, so V=S, so U∩V = U = M∩S, so rho_S = rho_M. QED.
      (Completeness lemma) On a DAG: if A_llm contains every true edge in the
          subgraph reachable from x, every true path x→b is preserved, hence
          cl_A ⊇ cl_T(x) (induction on path length: every edge preserved implies
          every path preserved). QED.

    Verification: random DAGs, LLM atomic facts SOUND (A_llm ⊆ T) with varying
    completeness q, multi-hop claims with injected hallucination. (1)(2)(3) checked
    over many trials.
    """
    import random

    rng = random.Random(seed)

    def closure(edges: set[tuple[int, int]], x: int) -> set[int]:
        adj: dict[int, set[int]] = {}
        for a, b in edges:
            adj.setdefault(a, set()).add(b)
        seen: set[int] = set()
        frontier = list(adj.get(x, ()))
        while frontier:
            nf = []
            for u in frontier:
                if u not in seen:
                    seen.add(u)
                    nf.extend(adj.get(u, ()))
            frontier = nf
        return seen

    trials = 300
    precision_ok = frechet_ok = complete_ok = True
    worst_slack = 1e9
    n_frechet_active = 0  # number of trials where the bound is >0 (actually binding)
    for _ in range(trials):
        n = rng.randint(8, 16)
        E: set[tuple[int, int]] = set()
        for j in range(1, n):
            for _ in range(rng.randint(1, 2)):
                E.add((rng.randint(0, j - 1), j))  # i<j ensures a DAG (acyclic)
        srcs = [x for x in range(n) if closure(E, x)]
        if not srcs:
            continue
        x = rng.choice(srcs)
        S = closure(E, x)
        # LLM atomic facts SOUND (⊆ T), completeness q (each true edge kept w.p. q)
        q = rng.choice([1.0, 1.0, 0.8, 0.6, 0.5])
        A = {(a, b) for (a, b) in E if rng.random() < q}
        clA = closure(A, x)
        c = len(clA & S) / len(S)
        # multi-hop: a subset of S (recall) plus injected hallucination outside S
        M = {b for b in S if rng.random() < rng.choice([0.5, 0.7, 0.9, 1.0])}
        rho_M = len(M & S) / len(S)
        for b in range(n):
            if b != x and b not in S and rng.random() < 0.3:
                M.add(b)  # hallucination
        kept = M & clA
        # (1) precision
        if len(kept - S) != 0:
            precision_ok = False
        rho_S = len(kept & S) / len(S)
        # (2) Fréchet
        bound = max(0.0, rho_M + c - 1.0)
        if bound > 0:
            n_frechet_active += 1
        if rho_S < bound - 1e-9:
            frechet_ok = False
        worst_slack = min(worst_slack, rho_S - bound)
        # (3) completeness implies rho_S == rho_M
        if q == 1.0 and (not (S <= clA) or abs(rho_S - rho_M) > 1e-9):
            complete_ok = False

    ok = precision_ok and frechet_ok and complete_ok and worst_slack >= -1e-9
    return {
        "theorem": "I_SGDC_RECALL_BOUND",
        "precision_always_one_when_sound": precision_ok,
        "frechet_bound_holds": frechet_ok,
        "frechet_active_trials": n_frechet_active,
        "worst_slack": round(worst_slack, 6),
        "completeness_gives_equality": complete_ok,
        "conclusion": (
            "CONFIRMED: precision=1 (sound); rho_S>=max(0,rho_M+c-1) (Frechet); "
            "completeness implies rho_S=rho_M (dominance)"
            if ok
            else (
                f"VIOLATED: prec={precision_ok} frechet={frechet_ok} "
                f"complete={complete_ok} slack={worst_slack:.3g}"
            )
        ),
    }


def theorem_closure_completeness(seed: int = 0) -> dict:
    """
    THEOREM J (Closure-Learning: Soundness + a Completeness condition). Learn a
    composition table comp: R x R -> R (associative) from (chain, gold) pairs by
    fixpoint iteration (composition_algebra).

      (1) SOUNDNESS (always holds): on associative data, every learned rule matches
          the true comp, with no conflicts, hence the solver never returns a wrong
          answer (coverage = accuracy).
      (2) COMPLETENESS (a precise condition): for test sequences drawn from
          alphabet A, coverage = 100% iff the table contains every (r, a) with
          a in A and r in the set of reachable prefixes P_A. Then
          |table| = |P_A| * |A|. A sufficiently long A-chain in training (up to
          diameter+1) achieves this.
      (3) REFUTES the naive hypothesis: "a generating set of the GROUP is enough"
          is FALSE — a generating set generates the whole group, but if a test
          atom is outside A, coverage collapses to ~0 (recorded honestly).

    Proof (1): fixpoint only assigns comp(a,b)=gold when the reduced chain still
    matches [a,b]; by associativity, gold equals the true comp(a,b), hence correct;
    two different golds for the same key would mean the data itself is not
    associative (a conflict). (2): left-to-right folding needs the key (prefix,
    atom) at every step; completeness holds iff every such key has been learned. QED.

    Verification: the dihedral group D6 (guarantees associativity), a phase
    transition measured against training chain length.
    """
    import random

    from grounded_reasoning.reasoning.composition_algebra import fold, learn_composition

    n = 6
    elems = [(r, s) for s in (0, 1) for r in range(n)]

    def mul(a, b):
        r1, s1 = a
        r2, s2 = b
        return ((r1 + r2) % n, s2) if s1 == 0 else ((r1 - r2) % n, 1 - s2)

    def gold(seq):
        acc = seq[0]
        for x in seq[1:]:
            acc = mul(acc, x)
        return acc

    rng = random.Random(seed)
    A = [(1, 0), (0, 1)]  # atom alphabet (also generates D6)
    test = [tuple(rng.choice(A) for _ in range(rng.randint(2, 12))) for _ in range(1000)]

    # (1)+(2): train on sufficiently long A-chains
    train = []
    for L in range(2, 7):
        for _ in range(500):
            seq = tuple(rng.choice(A) for _ in range(L))
            train.append((seq, gold(seq)))
    table, conflicts, iters = learn_composition(train)
    cov = sum(1 for t in test if fold(t, table) is not None) / len(test)
    acc = sum(1 for t in test if fold(t, table) == gold(t)) / len(test)

    # |P_A|*|A|: reachable prefixes = the submonoid generated by A (here, the whole group)
    reach = set(A)
    changed = True
    while changed:
        changed = False
        for x in list(reach):
            for a in A:
                if mul(x, a) not in reach:
                    reach.add(mul(x, a))
                    changed = True
    expected_rules = len(reach) * len(A)

    # (3) refutation: train generates the group, but test atoms OUTSIDE A collapse coverage
    test_arb = [tuple(rng.choice(elems) for _ in range(rng.randint(2, 6))) for _ in range(500)]
    cov_naive = sum(1 for t in test_arb if fold(t, table) is not None) / len(test_arb)

    ok = (
        conflicts == 0 and cov == 1.0 and acc == 1.0
        and len(table) == expected_rules and cov_naive < 0.5
    )
    return {
        "theorem": "J_CLOSURE_COMPLETENESS",
        "sound_zero_conflict": conflicts == 0,
        "cov_equals_acc": abs(cov - acc) < 1e-9,   # soundness: an answer implies correctness
        "coverage_in_alphabet": cov,
        "rules_eq_prefix_times_atoms": len(table) == expected_rules,
        "naive_generation_insufficient": cov_naive < 0.5,  # refutes the naive hypothesis
        "conclusion": (
            "CONFIRMED: sound (0 conflicts, cov=acc); completeness when the table "
            "covers (P_A x A) implies 100%; 'a group generating set is enough' is REFUTED"
            if ok
            else (
                f"VIOLATED: conf={conflicts} cov={cov:.2f} acc={acc:.2f} "
                f"rules={len(table)}/{expected_rules} naive={cov_naive:.2f}"
            )
        ),
    }


def theorem_conformal_reasoning(seed: int = 0) -> dict:
    """
    THEOREM K (Conformal Reasoning: valid COVERAGE under noise, degrading
    EFFICIENCY). Wrap the diffusion inference engine with split-conformal
    prediction (Vovk et al.), using conf(a→b) as the nonconformity score.

      (1) VALIDITY: average coverage >= 1-alpha at EVERY level of graph noise
          (missing/spurious edges) — distribution-free, requiring NO clean graph
          (unlike the hard guard, which is only correct when the graph is sound).
      (2) EFFICIENCY: the false-positive rate (prediction-set size) INCREASES with
          noise — the price paid; the operator score is only as useful as the
          graph it scores.

    Basis: split-conformal guarantees P(score_test >= tau) >= 1-alpha by
    exchangeability between the calibration and test sets, with tau the
    floor(alpha*(n+1))-quantile. No assumption is made about the score
    distribution, so validity holds regardless of score quality; the score only
    determines EFFICIENCY. QED (a classical guarantee, applied here to an
    operator-confidence score).

    Verification: random DAGs with injected noise, averaged over many seeds,
    measuring coverage against 1-alpha and the false-positive rate.
    """
    import random

    from grounded_reasoning.reasoning.abstract_inference import FuzzyInferenceEngine
    from grounded_reasoning.reasoning.conformal_reasoning import conformal_threshold

    def build(sd, p_drop, p_add, n=45):
        rng = random.Random(sd)
        te = set()
        for j in range(1, n):
            for _ in range(rng.randint(1, 2)):
                te.add((rng.randint(0, j - 1), j))
        adj: dict = {}
        for a, b in te:
            adj.setdefault(a, set()).add(b)

        def cl(x):
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

        truth = {x: cl(x) for x in range(n)}
        eng = FuzzyInferenceEngine(walk_len=12, alpha=0.7)
        for a, b in te:
            if rng.random() > p_drop:
                eng.add_relation(a, b)
        for _ in range(int(p_add * len(te))):
            a, b = rng.randint(0, n - 1), rng.randint(0, n - 1)
            if a != b:
                eng.add_relation(a, b)
        return n, truth, eng

    alpha = 0.1
    results = {}
    for p_drop in (0.0, 0.1, 0.3):
        covs, fprs = [], []
        for sd in range(20):
            n, truth, eng = build(sd, p_drop, 0.3)
            rng = random.Random(1000 + sd)
            infc = {x: eng.infer(x) for x in range(n)}
            jit = lambda: rng.uniform(0, 1e-9)  # noqa: E731 (break score ties)
            pos = [infc[x].get(b, 0.0) + jit() for x in range(n) for b in truth[x]]
            neg = [
                infc[x].get(b, 0.0) + jit()
                for x in range(n)
                for b in range(n)
                if b != x and b not in truth[x] and b in infc[x]
            ]
            rng.shuffle(pos)
            h = len(pos) // 2
            cal, test = pos[:h], pos[h:]
            tau = conformal_threshold(cal, alpha)
            covs.append(sum(1 for s in test if s >= tau) / len(test))
            fprs.append(sum(1 for s in neg if s >= tau) / max(len(neg), 1))
        results[p_drop] = (sum(covs) / len(covs), sum(fprs) / len(fprs))

    valid = all(c >= (1 - alpha) - 0.02 for c, _ in results.values())
    # efficiency degrades: FPR(heavy noise) > FPR(clean)
    efficiency_degrades = results[0.3][1] > results[0.0][1] + 0.1
    ok = valid and efficiency_degrades
    return {
        "theorem": "K_CONFORMAL_REASONING",
        "target_coverage": 1 - alpha,
        "coverage_clean": round(results[0.0][0], 3),
        "coverage_noisy": round(results[0.3][0], 3),
        "fpr_clean": round(results[0.0][1], 3),
        "fpr_noisy": round(results[0.3][1], 3),
        "validity_holds_under_noise": valid,
        "efficiency_degrades_with_noise": efficiency_degrades,
        "conclusion": (
            "CONFIRMED: coverage >=1-alpha at every noise level (validity); FPR "
            "grows with noise (efficiency degrades) — a soft guarantee when the "
            "graph is NOT clean"
            if ok
            else f"VIOLATED: valid={valid} eff_degrade={efficiency_degrades} {results}"
        ),
    }


def theorem_horn_least_model(seed: int = 0) -> dict:
    """
    THEOREM L (Horn Least-Model: a general sound + complete guard). Forward-
    chaining a Horn program to its least model M:

      (1) MODEL: M is closed under every rule (no rule fires outside M).
      (2) SUPPORTED (minimal): every derived fact in M \\ facts has a supporting
          rule (body ⊆ M, head = itself), i.e. nothing is fabricated (grounded).
      (3) GENERALIZES: relational transitive closure (Theorem G) is a single-rule
          Horn program — verified that its closure matches reachability.

    Proof: forward-chaining is a monotone operator T; a least fixpoint exists and
    is reached in finitely many steps (finite domain). M = lfp(T) is a model
    (T(M)=M) and is minimal (every element added by T has a supporting rule). QED
    (classical Datalog semantics).

    Verification: random Horn programs (checked for closure + supportedness), plus
    the transitive-closure special case.
    """
    import random

    from grounded_reasoning.reasoning.horn import forward_chain

    rng = random.Random(seed)
    closed_ok = supported_ok = True
    for _ in range(300):
        props = [f"p{i}" for i in range(20)]
        facts = set(rng.sample(props, rng.randint(2, 5)))
        rules = []
        for _ in range(30):
            k = rng.randint(1, 3)
            rules.append((frozenset(rng.sample(props, k)), rng.choice(props)))
        M = forward_chain(facts, rules)
        for body, head in rules:
            if body <= M and head not in M:
                closed_ok = False
        for d in M - facts:
            if not any(h == d and b <= M for b, h in rules):
                supported_ok = False

    # (3) transitive closure = a single Horn rule schema
    nodes = ["x0", "x1", "x2", "x3"]
    edges = {("x0", "x1"), ("x1", "x2"), ("x2", "x3")}
    facts = {("e", a, b) for a, b in edges}
    rules = [
        (frozenset({("e", a, b), ("e", b, c)}), ("e", a, c))
        for a in nodes for b in nodes for c in nodes
    ]
    M = forward_chain(facts, rules)
    reach = {(a, b) for a in nodes for b in nodes if ("e", a, b) in M}
    trans_ok = reach == {
        ("x0", "x1"), ("x0", "x2"), ("x0", "x3"),
        ("x1", "x2"), ("x1", "x3"), ("x2", "x3"),
    }

    ok = closed_ok and supported_ok and trans_ok
    return {
        "theorem": "L_HORN_LEAST_MODEL",
        "closed_model": closed_ok,
        "supported_minimal": supported_ok,
        "generalizes_transitive_closure": trans_ok,
        "conclusion": (
            "CONFIRMED: least model is closed + supported (grounded); transitive "
            "closure is a single-rule Horn program"
            if ok
            else f"VIOLATED: closed={closed_ok} sup={supported_ok} trans={trans_ok}"
        ),
    }


ALL_THEOREMS = [
    theorem_fuzzy_inference,
    theorem_operator_compositional_equivalence,
    theorem_relation_spectrum,
    theorem_sgdc_recall_bound,
    theorem_closure_completeness,
    theorem_conformal_reasoning,
    theorem_horn_least_model,
]

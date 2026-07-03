"""
General COMPOSITION algebra — learns the composition table of a monoid/group from
chain examples, via closure (fixpoint). An abstraction of CLUTRR's
`learn_table_closure` (DRY).

Given a token set R and an associative composition comp: R x R -> R. Observations
are chains (t1,...,tk) labeled with GOLD = the composition of the whole chain.
comp is learned by propagation:
  - direct pairs (length-2 chains) give comp(t1,t2)=gold;
  - longer chains reduced by already-known rules down to 2 elements infer the
    missing rule.
Iterate to a fixpoint. Because composition is ASSOCIATIVE, reducing at any
position is valid.
"""
from __future__ import annotations


def fold(seq: tuple, table: dict) -> object | None:
    """
    Reduce a chain using the composition table (CYK-style, any position). Returns
    None if a rule is missing OR the chain is empty (no implicit identity element —
    never fabricate a result).
    """
    if not seq:
        return None
    spans = list(seq)
    while len(spans) > 1:
        reduced = False
        for i in range(len(spans) - 1):
            key = (spans[i], spans[i + 1])
            if key in table:
                spans[i:i + 2] = [table[key]]
                reduced = True
                break
        if not reduced:
            return None
    return spans[0]


def learn_composition(chains: list[tuple[tuple, object]], max_iter: int = 1000):
    """
    Learn the table comp[(a,b)] -> c from (chain, gold) pairs by fixpoint.
    Returns (table, conflicts, iters). conflicts > 0 means the data is NOT
    associative/consistent.
    """
    table: dict[tuple, object] = {}
    conflicts = 0
    for it in range(1, max_iter + 1):
        changed = False
        for seq, gold in chains:
            spans = list(seq)
            # reduce as far as possible using the rules learned so far
            i = 0
            while len(spans) > 2 and i < len(spans) - 1:
                key = (spans[i], spans[i + 1])
                if key in table:
                    spans[i:i + 2] = [table[key]]
                    i = 0
                else:
                    i += 1
            if len(spans) == 2:
                key = (spans[0], spans[1])
                if key not in table:
                    table[key] = gold
                    changed = True
                elif table[key] != gold:
                    conflicts += 1
        if not changed:
            return table, conflicts, it
    return table, conflicts, max_iter

"""Method registry.

One name per row of the comparison matrix (paper §6.7, §7.1). The runner only
ever sees names, so adding a method means adding a module and one entry here.
"""

from __future__ import annotations

from eval.methods import (
    anubis,
    comboo,
    factorial,
    proposed,
    random_search,
    range_aware_tb,
    range_product,
    scbo,
    standard_bo,
)

#: How many multi-start points the acquisition optimizer gets.
#:
#: The production pipeline uses 128. Restart count dominates cost -- it is linear
#: in the number of scipy L-BFGS solves per iteration -- so the sweep sets it
#: once here for every method alike. Whatever it is, it must be identical across
#: methods, or the comparison measures compute rather than acquisition.
DEFAULT_NUM_RESTARTS = 128

#: name -> factory taking the restart count
REGISTRY: dict[str, callable] = {
    "M0_random": lambda n: random_search.make("M0_random"),
    "M1_qehvi_extremum": lambda n: standard_bo.make("M1_qehvi_extremum", num_restarts=n),
    "M2_target_distance": lambda n: standard_bo.make("M2_target_distance", num_restarts=n),
    "M3_range_prob": lambda n: proposed.make("M3_range_prob", num_restarts=n),
    "M4_range_prob_constraints": lambda n: proposed.make(
        "M4_range_prob_constraints", num_restarts=n
    ),
    "M5_full": lambda n: proposed.make("M5_full", num_restarts=n),
    # The four cells the cumulative ladder skips, completing the C1/C2/C3
    # factorial. M13 keeps C1; the other three switch it off, which needs the
    # care documented in `eval/methods/factorial.py` -- done naively they become
    # silent duplicates of M1.
    "M13_range_prob_starts": lambda n: proposed.make(
        "M13_range_prob_starts", num_restarts=n
    ),
    "M14_constraints_only": lambda n: factorial.make(
        "M14_constraints_only", num_restarts=n
    ),
    "M15_starts_only": lambda n: factorial.make("M15_starts_only", num_restarts=n),
    "M16_constraints_starts": lambda n: factorial.make(
        "M16_constraints_starts", num_restarts=n
    ),
    # C4: the full method plus Anubis's feasibility signal, grafted onto the
    # candidate ranking (C4a) and onto the restart filter (C4b). Separate entries
    # because the two can disagree -- C4b competes with C3 for the same restart
    # pool, C4a does not. Shares one classifier with `M9_anubis`
    # (`src/utils/feasibility.py`) so the comparison isolates where the signal is
    # applied, not which classifier produced it.
    "M10_full_feasw": lambda n: proposed.make("M10_full_feasw", num_restarts=n),
    "M11_full_feasf": lambda n: proposed.make("M11_full_feasf", num_restarts=n),
    "M12_full_feas": lambda n: proposed.make("M12_full_feas", num_restarts=n),
    # §4.2 specifies the product of range probabilities; the shipped multi-output
    # path maximizes hypervolume over the probability vector instead. These four
    # implement the specified form -- directly and wrapped in qEI, with C2/C3 off
    # and on -- so aggregation, acquisition form, and constraint interaction can
    # be separated. See `eval/methods/range_product.py`.
    "M3p_range_prob_product": lambda n: range_product.make(
        "M3p_range_prob_product", num_restarts=n
    ),
    "M3q_range_prob_product_ei": lambda n: range_product.make(
        "M3q_range_prob_product_ei", num_restarts=n
    ),
    "M5p_full_product": lambda n: range_product.make("M5p_full_product", num_restarts=n),
    "M5q_full_product_ei": lambda n: range_product.make(
        "M5q_full_product_ei", num_restarts=n
    ),
    # Two box-to-ball mappings, because neither is canonical and the choice
    # would otherwise decide the comparison on its own. equal_volume leads;
    # inscribed is the containment-guaranteed sensitivity check.
    "M6_range_aware_tb": lambda n: range_aware_tb.make("M6_range_aware_tb", num_restarts=n),
    "M6b_range_aware_tb_inscribed": lambda n: range_aware_tb.make(
        "M6b_range_aware_tb_inscribed", num_restarts=n
    ),
    # Single-output suites only: the ball becomes the window exactly, so this is
    # the range probability without C1's improvement hinge. It isolates the
    # plateau -- see `range_aware_tb.MAPPINGS`. Raises on a multi-output task.
    "M6c_range_aware_tb_box": lambda n: range_aware_tb.make(
        "M6c_range_aware_tb_box", num_restarts=n
    ),
    # §6.5 general constrained BO. Multi-output only -- it raises on a
    # single-output task rather than degenerating quietly.
    "M8_comboo": lambda n: comboo.make("M8_comboo", num_restarts=n),
    # Unknown-feasibility-aware baseline (Anubis, Hickman et al. 2025) --
    # HOIP's own source paper's algorithm, not just its benchmark task.
    # Catalogue-only: needs `restart_pool` (enumerable design space), which
    # only HOIP currently supplies.
    "M9_anubis": lambda n: anubis.make("M9_anubis", num_restarts=n),
    # Trust-region constrained BO (Eriksson & Poloczek 2021). Single-output
    # only -- it raises on a multi-output task rather than degenerating
    # quietly, the mirror image of M8_comboo's own output-arity guard.
    "M7_scbo": lambda n: scbo.make("M7_scbo", num_restarts=n),
}

#: The C1/C2/C3 ablation ladder, in the order the paper's §7.1 table presents it.
ABLATION_LADDER = [
    "M1_qehvi_extremum",
    "M3_range_prob",
    "M4_range_prob_constraints",
    "M5_full",
]

#: The full 2x2x2 factorial over (C1, C2, C3), keyed by which components are on.
#: The ladder above is the diagonal of this; the other four cells are what
#: separate "C2/C3 amplify C1" from "C2/C3 do the work".
FACTORIAL: dict[tuple[bool, bool, bool], str] = {
    (False, False, False): "M1_qehvi_extremum",
    (True, False, False): "M3_range_prob",
    (True, True, False): "M4_range_prob_constraints",
    (True, False, True): "M13_range_prob_starts",
    (True, True, True): "M5_full",
    (False, True, False): "M14_constraints_only",
    (False, False, True): "M15_starts_only",
    (False, True, True): "M16_constraints_starts",
}


def make(name: str, num_restarts: int = DEFAULT_NUM_RESTARTS):
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    method = REGISTRY[name](num_restarts)
    # Result directories are keyed by registry name while tables read `method`
    # off the metrics; a mismatch splits one method across two labels.
    if getattr(method, "name", name) != name:
        raise AssertionError(
            f"registry key {name!r} does not match method name {method.name!r}"
        )
    return method

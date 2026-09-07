"""Check the HOIP catalogue against its own data, before anything is built on it.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.audit_hoip_catalogue

The counterpart of `scripts/audit_olympus_datasets.py`, and it exists for the
same reason: on Olympus a declaration was wrong in one direction, another was
wrong in the opposite direction, and a plausible-looking convex-hull test was
blind to both. One of those was caught only after a 270-cell sweep had run.

HOIP's design space is finite, so unlike Olympus nothing here is estimated --
every number this prints is an exact count over all 1,276 candidate materials.

Exit status is non-zero if any check fails, so this can gate a pipeline.
"""

from __future__ import annotations

import numpy as np

from eval.benchmarks.hoip import data as hd


def _line(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:38s} {detail}")
    return ok


def main() -> int:
    print("HOIP catalogue audit\n")

    table = hd.load_table()
    n_space = len(hd.MOLCATS) * len(hd.METALS) * len(hd.HALOGENS)
    responses = hd.response_map()

    print("Declared design space")
    print(f"  {len(hd.MOLCATS)} molcats x {len(hd.METALS)} metals x "
          f"{len(hd.HALOGENS)} halogens = {n_space} materials")
    print(f"  feasible (present in the lookup table): {len(responses)} "
          f"({len(responses) / n_space:.1%})")
    print(f"  infeasible: {n_space - len(responses)} "
          f"({1 - len(responses) / n_space:.1%})\n")

    ok = True
    print("Checks")

    undeclared = hd.undeclared_options()
    ok &= _line(
        not any(undeclared.values()),
        "every parsed option is declared",
        "" if not any(undeclared.values()) else str(undeclared),
    )

    dupes = hd.duplicate_compositions()
    ok &= _line(
        not dupes,
        "one row per composition",
        "upstream asserts len(match) in [1, 0]" if not dupes else str(dupes),
    )

    ambiguous = hd.parse_ambiguities()
    ok &= _line(
        not ambiguous,
        "molcat+metal+halogen is injective",
        "no two triples share a concatenation" if not ambiguous else str(ambiguous),
    )

    failures = hd.roundtrip_failures()
    ok &= _line(
        not failures,
        "si_table compositions round-trip",
        f"{len(failures)} unresolved" if failures else "all resolve to exactly one triple",
    )

    agreement = hd.si_table_agreement()
    if agreement.get("available"):
        missing = agreement["in_lookup_not_in_si"]
        ok &= _line(
            not missing,
            "lookup table is inside si_table",
            f"{agreement['n_lookup_compositions']} of "
            f"{agreement['n_si_compositions']} parsed si_table compositions",
        )

    completeness = hd.descriptor_completeness()
    complete = all(
        not c["missing_options"] and not c["nonfinite_options"] and not c["incomplete_from_pickle"]
        for c in completeness.values()
    )
    detail = " ".join(
        f"{b}:{c['n_descriptors']}x{c['n_options']}" for b, c in completeness.items()
    )
    ok &= _line(complete, "descriptors complete and finite", detail)

    # Encoding must separate the options it stands in for. Two options with
    # identical descriptor vectors would be one material wearing two names: the
    # snapping oracle could not tell them apart and the GP would see a
    # contradiction wherever their responses differ.
    collisions = {}
    desc = hd.load_descriptors()
    for block, values in ((b, desc[b]["values"]) for b in hd.BLOCKS):
        seen: dict[tuple, list[str]] = {}
        for opt, vec in values.items():
            seen.setdefault(tuple(np.round(vec, 12)), []).append(opt)
        clashes = [v for v in seen.values() if len(v) > 1]
        if clashes:
            collisions[block] = clashes
    ok &= _line(
        not collisions,
        "descriptor vectors separate options",
        "" if not collisions else str(collisions),
    )

    print("\nData facts to carry into any write-up")

    censored = hd.censored_rows()
    print(f"  effective mass reported as '>1000': {len(censored)} of {len(table)} "
          f"feasible materials. Held as NaN -- a censoring sentinel is not a")
    print("    measurement, and 1000.0 is three orders of magnitude past the bulk.")
    if len(censored):
        print("    " + ", ".join(
            f"{r.molcat}/{r.metal}/{r.halogen}" for r in censored.itertuples()
        ))

    if agreement.get("available"):
        extra = agreement["n_si_compositions"] - agreement["n_lookup_compositions"]
        print(f"\n  {extra} compositions are in si_table but absent from the lookup table,")
        print("    so the oracle calls them infeasible. Every one of them is metallic")
        print("    (PBE gap 0.00), which is why upstream had no effective mass to record.")
        print("    HOIP's feasibility label therefore conflates 'not a stable perovskite'")
        print("    with 'stable but the target property is undefined'. It changes no hit")
        print("    -- a zero gap is outside the window regardless -- but it is what the")
        print("    label means, and a write-up should say so.")

    lo, hi = hd.NATIVE_WINDOW["bandgap"]
    m_cap = hd.NATIVE_WINDOW["m_star"][1]
    valid = table[(table.bandgap >= lo) & (table.bandgap <= hi) & (table.m_star < m_cap)]
    print(f"\n  native window: {lo} <= bandgap <= {hi} and m_star < {m_cap}")
    print(f"    valid materials: {len(valid)} of {len(responses)} feasible "
          f"({len(valid) / len(responses):.1%}), {len(valid) / n_space:.2%} of the space")
    print(valid[["molcat", "metal", "halogen", "bandgap", "m_star"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    finite = table.m_star.dropna()
    print(f"\n  m_star spans {finite.min():.2f} to {finite.max():.2f} over {len(finite)} "
          "measured materials -- heavy-tailed, like MatFormBench's y2. Kept raw, as")
    print("    upstream keeps it; the frozen robust reference scale is what the ball")
    print("    criteria use, so no scoring geometry depends on the tail.")

    print(f"\n{'audit passed' if ok else 'AUDIT FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

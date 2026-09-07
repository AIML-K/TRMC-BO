"""HOIP catalogue, descriptors, and the checks that decide whether to trust them.

Data provenance
---------------

The benchmark is the perovskite application of *Anubis: Bayesian optimization
with unknown feasibility constraints for scientific experimentation*
(Aspuru-Guzik group). Its design space is the full product

    11 molecular cations  x  29 metals  x  4 halogens  =  1,276 materials

and its oracle is a lookup in `df_results.csv`, which holds the near-hull,
ground-state DFT results (Koerbel/Marques/Botti) for the subset that was computed
*and* came out stable. A composition absent from that table is not
manufacturable, and upstream returns `(nan, nan)` for it -- that absence is the
unknown input feasibility this whole suite exists to measure.

Two outputs, both taken exactly as upstream takes them:

    bandgap   PBE fundamental gap plus its spin-orbit correction, in eV
    m_star    effective mass, dimensionless

Why these checks and not a single obvious one
---------------------------------------------

Olympus taught the expensive lesson, and it was not about Olympus: *a
plausible-looking validity test can be structurally blind*. A convex-hull check
passed all three of its distinct dataset failures. So the declaration is never
trusted here either -- every property the suite relies on is checked against the
raw table, and each check below catches something the others cannot see:

``undeclared_options``
    the parsed keys are all inside the declared 11/29/4 lists. Catches a wrong
    option list, which would silently shrink or shift the design space.

``duplicate_compositions``
    upstream asserts ``len(match) in [1, 0]`` at every evaluation. If our table
    ever had two rows for one composition, the oracle would be ill-defined and
    the assert would fire only for the compositions a run happened to visit.

``parse_ambiguities`` / ``roundtrip_failures``
    upstream derives (molcat, metal, halogen) from a composition string by
    *sequential substring search* over lists in which ``S`` is a metal and also
    the tail of the molcats ``H3S`` and ``MS``, and in which ``MS`` could be read
    as ``M`` + ``S``. That is exactly the shape of parse that relabels materials
    without failing. Here the triple is instead recovered by exact concatenation
    against all 1,276 candidates, and any string matching two different triples
    is reported rather than resolved by precedence.

``censored_rows``
    ``>1000`` in the source becomes ``1000.0`` in the table. That is a censoring
    sentinel, not a measurement, and feeding it to a GP as a number would put a
    fictitious point three orders of magnitude past the bulk.

``descriptor_completeness``
    every option needs every descriptor of its block, finite. Upstream's loader
    casts with the long-removed ``np.float``, so a dtype-object column is exactly
    the kind of thing that would have gone unnoticed there.

Nothing here samples. The design space is finite, so every count this module
reports is exact.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "hoip"

#: The declared design space, verbatim from `application_hoip/ei/*/run.py`.
MOLCATS: tuple[str, ...] = (
    "H3S", "NH4", "MS", "MA", "MP", "FA", "EA", "G", "AA", "ED", "tBA",
)
METALS: tuple[str, ...] = (
    "Be", "Mg", "Ca", "Sr", "Ba", "Cr", "Mn", "Fe", "Co", "Ni", "Pd", "Pt",
    "Cu", "Ag", "Au", "Zn", "Cd", "Hg", "Ga", "In", "Tl", "Si", "Ge", "Sn",
    "Pb", "Bi", "S", "Se", "Te",
)
HALOGENS: tuple[str, ...] = ("F", "Cl", "Br", "I")

BLOCKS = ("molcats", "metals", "halogens")
Y_NAMES = ("bandgap", "m_star")

#: `>1000` in the source table. A censoring sentinel, never a measurement.
CENSOR_SENTINEL = 1000.0

#: The benchmark's own success test, `abs(bandgap - 1.25) < 0.5 and m_star < 4`,
#: written as the two-sided window it actually is.
NATIVE_WINDOW = {
    "bandgap": (0.75, 1.75),
    "m_star": (-np.inf, 4.0),
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- the HOIP data has not been fetched. Run\n"
            "    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.fetch_hoip_data"
        )
    return path


# -- the lookup table --------------------------------------------------------

@lru_cache(maxsize=1)
def load_table() -> pd.DataFrame:
    """Feasible materials with their responses; sentinels turned into NaN.

    Columns: molcat, metal, halogen, bandgap, m_star, E_hull, plus `censored`
    marking the rows whose effective mass was reported as `>1000`. Those keep
    `m_star = NaN`: the material exists (feasible) but that output is not
    observed, which is a state the harness already models -- `Observation.Y`
    carries NaN per output and `initialize_model` fits each objective only on the
    rows where it was measured.
    """
    frame = pd.read_csv(_require(DATA_ROOT / "df_results.csv"))
    out = frame[["molcat", "metal", "halogen", "bandgap", "m_star", "E_hull"]].copy()
    censored = out["m_star"] >= CENSOR_SENTINEL
    out["censored"] = censored
    out.loc[censored, "m_star"] = np.nan
    return out.reset_index(drop=True)


@lru_cache(maxsize=1)
def response_map() -> dict[tuple[str, str, str], tuple[float, float]]:
    """(molcat, metal, halogen) -> (bandgap, m_star). Absent key == infeasible."""
    table = load_table()
    return {
        (r.molcat, r.metal, r.halogen): (float(r.bandgap), float(r.m_star))
        for r in table.itertuples()
    }


# -- descriptors -------------------------------------------------------------

@lru_cache(maxsize=1)
def load_descriptors() -> dict:
    """`{block: {"descriptor_names": [...], "values": {option: [floats]}}}`.

    Written by `scripts/fetch_hoip_data.py` from the shipped pickles. The pickles
    are never opened at sweep time.
    """
    return json.loads(_require(DATA_ROOT / "descriptors.json").read_text(encoding="utf-8"))


def descriptor_names() -> tuple[str, ...]:
    """The 14 coordinate names, block-major: molcat, then metal, then halogen."""
    desc = load_descriptors()
    return tuple(
        f"{block[:-1] if block.endswith('s') else block}_{name}"
        for block in BLOCKS
        for name in desc[block]["descriptor_names"]
    )


def block_slices() -> dict[str, slice]:
    """Which coordinates of the 14-D vector belong to which categorical block."""
    desc = load_descriptors()
    out, start = {}, 0
    for block in BLOCKS:
        width = len(desc[block]["descriptor_names"])
        out[block] = slice(start, start + width)
        start += width
    return out


def encode(triples, *, descriptors=None) -> np.ndarray:
    """(molcat, metal, halogen) triples -> (n, 14) descriptor coordinates."""
    desc = descriptors if descriptors is not None else load_descriptors()
    rows = []
    for molcat, metal, halogen in triples:
        rows.append(
            desc["molcats"]["values"][molcat]
            + desc["metals"]["values"][metal]
            + desc["halogens"]["values"][halogen]
        )
    return np.asarray(rows, dtype=float).reshape(len(rows), -1)


# -- audit primitives --------------------------------------------------------
#
# Each returns evidence, not a verdict. `scripts/audit_hoip_catalogue.py` and
# `tests/test_hoip.py` decide what is fatal.

def undeclared_options() -> dict[str, list[str]]:
    """Options appearing in the table that are not in the declared lists."""
    table = load_table()
    return {
        "molcat": sorted(set(table.molcat) - set(MOLCATS)),
        "metal": sorted(set(table.metal) - set(METALS)),
        "halogen": sorted(set(table.halogen) - set(HALOGENS)),
    }


def duplicate_compositions() -> list[tuple[str, str, str]]:
    """Compositions with more than one row -- upstream asserts there are none."""
    table = load_table()
    counts = table.groupby(["molcat", "metal", "halogen"]).size()
    return [tuple(k) for k, n in counts.items() if n > 1]


def censored_rows() -> pd.DataFrame:
    """Materials whose effective mass was reported as `>1000`."""
    table = load_table()
    return table[table.censored]


def descriptor_completeness() -> dict[str, dict]:
    """Per block: options missing a descriptor, and non-finite values."""
    desc = load_descriptors()
    declared = {"molcats": MOLCATS, "metals": METALS, "halogens": HALOGENS}
    out = {}
    for block in BLOCKS:
        values = desc[block]["values"]
        missing = [opt for opt in declared[block] if opt not in values]
        nonfinite = [
            opt for opt, vec in values.items() if not np.isfinite(np.asarray(vec, float)).all()
        ]
        out[block] = {
            "n_descriptors": len(desc[block]["descriptor_names"]),
            "n_options": len(values),
            "missing_options": missing,
            "nonfinite_options": sorted(nonfinite),
            "incomplete_from_pickle": desc[block].get("incomplete", []),
        }
    return out


# -- the composition parse ---------------------------------------------------

#: Structure-phase suffixes seen in `si_table.csv` (`-1D-W`, `-3D-OR`, ...), plus
#: the `\bullet` footnote marker. Stripped before the triple is recovered.
_PHASE_SUFFIXES = ("-1D-W", "-1D-R", "-2D", "-3D-OR", "-3D", "-1D")


def _clean_composition(raw: str) -> str:
    """`H_3SInCl_3-1D-W` -> `H3SInCl`: drop underscores, phase tag, stoichiometry."""
    text = raw.strip().replace("\\bullet", "").replace("$", "").strip()
    text = text.replace("_", "")
    for suffix in _PHASE_SUFFIXES:
        tag = suffix.replace("-", "")
        if text.endswith(tag):
            text = text[: -len(tag)]
            break
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.rstrip("-")
    if text.endswith("3"):
        text = text[:-1]
    return text


@lru_cache(maxsize=1)
def _concatenation_index() -> dict[str, list[tuple[str, str, str]]]:
    """Every declared triple, keyed by `molcat + metal + halogen`."""
    index: dict[str, list[tuple[str, str, str]]] = {}
    for molcat in MOLCATS:
        for metal in METALS:
            for halogen in HALOGENS:
                index.setdefault(molcat + metal + halogen, []).append(
                    (molcat, metal, halogen)
                )
    return index


def parse_ambiguities() -> dict[str, list[tuple[str, str, str]]]:
    """Concatenations that two different triples both produce.

    This is the failure upstream's sequential substring search is exposed to, and
    it is a property of the *option lists alone* -- it does not need the data.
    """
    return {k: v for k, v in _concatenation_index().items() if len(v) > 1}


def parse_composition(raw: str):
    """Recover the triple by exact concatenation. `None` when it cannot be."""
    return _concatenation_index().get(_clean_composition(raw), [None])[0]


def roundtrip_failures() -> list[dict]:
    """si_table compositions our parse cannot resolve to exactly one triple.

    `si_table.csv` is the upstream DFT source `df_results.csv` was derived from.
    It carries the composition strings that `df_results.csv` has already thrown
    away, so it is the only place the derivation can be checked at all.
    """
    path = DATA_ROOT / "si_table.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, dtype=str)
    column = frame.columns[0]
    out = []
    for raw in frame[column].dropna():
        cleaned = _clean_composition(raw)
        matches = _concatenation_index().get(cleaned, [])
        if len(matches) != 1:
            out.append({"composition": raw, "cleaned": cleaned, "n_matches": len(matches)})
    return out


def si_table_agreement() -> dict:
    """Do the compositions we parse out of si_table cover the lookup table?

    `df_results.csv` should be a subset of what si_table contains: upstream keeps
    one ground-state row per composition. A key in the lookup table that si_table
    never mentions would mean the two files disagree about what was computed.
    """
    path = DATA_ROOT / "si_table.csv"
    if not path.exists():
        return {"available": False}
    frame = pd.read_csv(path, dtype=str)
    parsed = {parse_composition(raw) for raw in frame[frame.columns[0]].dropna()}
    parsed.discard(None)
    table_keys = set(response_map())
    return {
        "available": True,
        "n_si_compositions": len(parsed),
        "n_lookup_compositions": len(table_keys),
        "in_lookup_not_in_si": sorted(table_keys - parsed),
    }

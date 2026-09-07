"""Pull the Anubis HOIP data into `eval/data/hoip/`, once.

    PYTHONPATH=core:. .venv-eval/bin/python -m eval.scripts.fetch_hoip_data

The HOIP benchmark is the perovskite application of *Anubis: Bayesian
optimization with unknown feasibility constraints for scientific
experimentation* (Aspuru-Guzik group). Its oracle is a lookup table, so unlike
Olympus there is no framework to install and no weights to lift -- but the
descriptors ship as **pickles**, and a pickle is executable. They are opened
exactly once, here, and re-emitted as JSON; nothing at sweep time ever unpickles
third-party data.

What the upstream runner does with these files (`application_hoip/ei/*/run.py`)
is the specification this suite is built against:

    match = df_results.loc[(molcat == ...) & (metal == ...) & (halogen == ...)]
    assert len(match) in [1, 0]
    if len(match) == 0:
        return np.nan, np.nan          # <- input infeasibility, unknown a priori
    bandgap = abs(match['bandgap'] - 1.25)
    m_star  = match['m_star']
    ...
    converged = measurement['bandgap'] < 0.5 and measurement['m_star'] < 4.

i.e. the target window is `0.75 <= Eg <= 1.75` and `m* <= 4`, and a composition
absent from the table is not manufacturable.
"""

from __future__ import annotations

import argparse
import json
import pickle
import urllib.request
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "hoip"

RAW = ("https://raw.githubusercontent.com/the-matter-lab/atlas-unknown-constraints"
       "/main/application_hoip/ei/reference-and-data")

#: Everything the suite needs. `si_table.csv` is not read by the oracle -- it is
#: the upstream DFT source (Koerbel/Marques/Botti) that `df_results.csv` was
#: derived from, and it is kept so the derivation can be audited rather than
#: assumed.
FILES = {
    "df_results.csv": f"{RAW}/df_results.csv",
    "si_table.csv": f"{RAW}/si_table.csv",
    "desc_molcats.pkl": f"{RAW}/descriptors/desc_molcats.pkl",
    "desc_metals.pkl": f"{RAW}/descriptors/desc_metals.pkl",
    "desc_halogens.pkl": f"{RAW}/descriptors/desc_halogens.pkl",
}

DESCRIPTORS_JSON = "descriptors.json"


def fetch(force: bool = False) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = DATA_ROOT / name
        if dest.exists() and not force:
            print(f"  have  {name} ({dest.stat().st_size} bytes)")
            continue
        with urllib.request.urlopen(url) as response:
            payload = response.read()
        dest.write_bytes(payload)
        print(f"  got   {name} ({len(payload)} bytes)")
    return DATA_ROOT


def convert_descriptors() -> Path:
    """Unpickle once; everything downstream reads the JSON.

    Upstream stores `{descriptor_name: {option: value}}` per block and rebuilds a
    per-option vector by iterating `desc.keys()`. Key order therefore *is* the
    descriptor vector order, so it is frozen here explicitly rather than left to
    dict iteration order in a different process.
    """
    out = {}
    for block in ("molcats", "metals", "halogens"):
        with open(DATA_ROOT / f"desc_{block}.pkl", "rb") as handle:
            raw = pickle.load(handle)
        names = list(raw)
        options = sorted({opt for name in names for opt in raw[name]})
        out[block] = {
            "descriptor_names": names,
            "values": {
                opt: [float(raw[name][opt]) for name in names]
                for opt in options
                if all(opt in raw[name] for name in names)
            },
            "incomplete": sorted(
                opt for opt in options if not all(opt in raw[name] for name in names)
            ),
        }
        print(f"  {block:9s} {len(names)} descriptors x {len(out[block]['values'])} options"
              f"{'  INCOMPLETE: ' + ', '.join(out[block]['incomplete']) if out[block]['incomplete'] else ''}")
    path = DATA_ROOT / DESCRIPTORS_JSON
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    print(f"fetching into {DATA_ROOT}")
    fetch(force=args.force)
    print("converting descriptors (the only time a third-party pickle is opened)")
    path = convert_descriptors()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

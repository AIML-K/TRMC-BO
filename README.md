# TRMC-BO: Reproducibility Package

Code and experiments for **"TRMC-BO: Target-Range Bayesian Optimization under Multiple Constraints for Materials Formulation"** (*Digital Discovery*, RSC).

Three public benchmarks — **HOIP**, **Olympus**, **MatFormBench** — reproduced end to end: environment, data, and the full sweep/report pipeline.
The proprietary optimizer core ships as a compiled black box (see below); 
every number in the paper is reproducible from what's in this repo.

## Quickstart

```bash
docker build -f docker/Dockerfile -t trmcbo-reproduce .
docker run --rm trmcbo-reproduce                       # usage + status, all suites
docker run --rm trmcbo-reproduce hoip verify            # ~1 min
docker run --rm trmcbo-reproduce hoip sweep             # full campaign
docker run --rm trmcbo-reproduce olympus report
docker run --rm trmcbo-reproduce matformbench all
```

No manual environment setup, no dataset hunting: the image clones the two third-party benchmarks, builds both Python environments, and stages every dataset at build time. 
`verify` (audits, ablation-toggle checks, unit tests) finishes in about a minute per suite; 
`sweep`/`all` run the real campaigns and are resumable if interrupted.

Prefer bare metal? 
`eval/README.md` has the manual venv/clone instructions every `reproduce_*.sh` script also works from directly.

## Why part of this is a compiled binary

The optimizer itself (`core/`) is co-developed with an industrial partner under a pending patent and ships as `.so` binaries — no Python source, whole
`src/` tree compiled via Cython. 
This is not a black box wrapped around someone else's numbers: 
the same binary is what every result in the paper was produced with, and its outputs are verified bit-for-bit identical to the original source (see `build/build_core.py`'s header for how, and the paper's data availability statement for why proprietary code and formulation data can't ship as source).

Everything *around* the core — benchmarks, baselines, ablations, sweep harness, plotting — is plain, readable Python.

## Layout

```
core/                 compiled optimizer (.so), no source
eval/                  benchmarks, baselines, sweep harness, tests -- all open
docker/                Dockerfile + entrypoint
reproduce_hoip.sh
reproduce_olympus.sh
reproduce_matformbench.sh
```

Each `reproduce_<suite>.sh` is a self-documenting dispatcher:
`status | verify | sweep | report | ...` — run one with no argument for its own usage.

## Verified

All three suites: 
full test suite green, every ablation switch (which objective, which constraints, which restart filtering) confirmed to actually reach the optimizer rather than silently doing nothing, frozen target windows and catalogues reproduce against their committed values, and a full campaign re-run against the compiled core matches the original source-run trace byte-for-byte.

## Citation


## License

TBU

# Build context is reproduce/ itself, never the repo root:
#
#   docker build -f docker/Dockerfile -t trmcbo-reproduce .
#   (run from inside reproduce/, or: docker build -f reproduce/docker/Dockerfile reproduce/)
#
# This makes it structurally impossible for src/ (the proprietary source, which
# lives outside reproduce/ and is never staged here) to leak into the image --
# whatever ships in reproduce/ is exactly what a public checkout would contain.
#
# The proprietary method itself never gets compiled here: core/*.so is already
# a committed, built artifact (see build/build_core.py, which needs src/ and is
# irrelevant to this image). This Dockerfile only reproduces the *run*
# environment, not the build environment.

# ---------------------------------------------------------------------------
# Stage: fetch MatFormBench (third-party, public, unauthenticated clone).
# Discarded after its one file tree is copied into the runtime stage, so git
# and its own dependency chain never need to persist.
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS matformbench-fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/DeepVerse/MatFormBench.git /tmp/MatFormBench

# ---------------------------------------------------------------------------
# Stage: fetch Olympus (third-party, public, unauthenticated clone). Needed at
# *runtime*, not just to extract emulator weights: benchmarks/olympus/data.py
# reads measured-dataset CSVs straight out of this clone
# (eval/external/olympus/src/olympus/datasets/dataset_<name>/data.csv) for the
# pce10/oer_plate_3496 tasks, which the pre-committed emulator .npz weights
# alone do not cover.
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS olympus-fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/aspuru-guzik-group/olympus.git /tmp/olympus

# ---------------------------------------------------------------------------
# Stage: runtime (the final image)
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# procps: reproduce_*.sh's warn_if_busy() uses pgrep.
# libgomp1: runtime dependency of lightgbm/catboost (pulled in by MatFormBench's
# own requirements.txt).
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repro

COPY --from=matformbench-fetch /tmp/MatFormBench /repro/eval/external/MatFormBench
COPY --from=olympus-fetch /tmp/olympus /repro/eval/external/olympus

# /opt/venv-eval -- the shared experiment environment for all three suites'
# sweeps/reports. Order matters: MatFormBench's own (unpinned) requirements
# pull numpy 2.x; the override list below re-pins to 1.26.4, which the
# production GP stack needs. See eval/README.md's "Environment" section.
RUN python -m venv /opt/venv-eval
RUN /opt/venv-eval/bin/pip install --no-cache-dir --upgrade pip
# CPU-only index here too: MatFormBench's own requirements.txt pins an
# unversioned "torch", which would otherwise resolve a CUDA build from PyPI's
# default index before the pinned CPU-only reinstall below replaces it anyway.
RUN /opt/venv-eval/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r /repro/eval/external/MatFormBench/requirements.txt
# CPU-only torch build: a plain `pip install torch` pulls ~5-6GB of unused
# CUDA/nvidia-* wheels (this project only ever runs on CPU, matching the host
# .venv-eval and .venv-olympus setup). --extra-index-url still lets pip resolve
# the rest of the list (botorch, tabpfn, ...) from PyPI as normal.
RUN /opt/venv-eval/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.6.0 botorch==0.15.0 gpytorch==1.14 linear-operator==0.6 \
        drs==2.0.0 scikit-learn==1.6.1 scipy==1.15.2 numpy==1.26.4 pandas==2.2.3 \
        threadpoolctl tqdm pyarrow pyyaml pytest \
        tabpfn

# /opt/venv-olympus -- TensorFlow, used only by `verify_olympus_emulator.py`'s
# fidelity gate (`reproduce_olympus.sh verify`). Never touched at sweep/report
# time, so it never competes with the eval workers for threads; the extracted
# emulator weights themselves (eval/data/olympus/*.npz+json) are committed
# artifacts, copied in below with the rest of eval/ -- no re-extraction is
# needed to use or verify them. The olympus clone above is still required at
# sweep time regardless (measured-dataset CSVs for pce10/oer_plate_3496).
RUN python -m venv /opt/venv-olympus
RUN /opt/venv-olympus/bin/pip install --no-cache-dir --upgrade pip
RUN /opt/venv-olympus/bin/pip install --no-cache-dir \
        "tensorflow-cpu==2.15.*" "tensorflow-probability==0.23.0" numpy==1.26.4

COPY core/ /repro/core/
COPY eval/ /repro/eval/
COPY reproduce_hoip.sh reproduce_olympus.sh reproduce_matformbench.sh /repro/
COPY docker/entrypoint.sh /repro/docker/entrypoint.sh
RUN chmod +x /repro/reproduce_hoip.sh /repro/reproduce_olympus.sh \
              /repro/reproduce_matformbench.sh /repro/docker/entrypoint.sh

ENV PYTHONPATH=core:. \
    PY=/opt/venv-eval/bin/python \
    PY_TF=/opt/venv-olympus/bin/python

ENTRYPOINT ["/repro/docker/entrypoint.sh"]
CMD ["status"]

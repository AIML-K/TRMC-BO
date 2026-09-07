"""Compile `src/` into the black-box `reproduce/core/` package.

Cythonizes every `.py` file under `src/` in "pure Python" mode (no `.pyx`
rewrite needed -- `src/` has no `cdef`/`cpdef` anywhere) and builds each into a
`.so` extension module under `reproduce/core/`, mirroring `src/`'s directory
layout. Only the compiled binaries land in `reproduce/core/` -- no `.py`
source for any compiled module is copied there.

`binding=True` keeps plain `def` functions as regular Python-level callables
resolved through the module's globals at call time (not statically bound at
compile time), which is what lets `eval/scripts/verify_variants.py` and
`eval/scripts/acquisition_plateau.py` monkeypatch module-level functions
(`bo.optimize_acqf_on_filtered_points = ...`) and have `run_mobo`'s internal
call sites actually pick up the replacement. `embedsignature=True` keeps
`inspect.signature(...)` working, which `eval/tests/test_src_hooks.py` relies
on directly.

Run with the CPython 3.10 build venv (`.venv-build`), matching `.venv-eval`'s
ABI, since `.so` extensions are locked to a CPython minor version:

    .venv-build/bin/python reproduce/build/build_core.py
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CORE = ROOT / "reproduce" / "core"

# Cython embeds each extension's source *path* as a literal string (for
# tracebacks), and this checkout's own directory happens to be named after the
# partner company -- so building in place would bake "KOLON-BO" into every
# compiled .so regardless of what any source file says. Stage in a neutrally
# named system temp dir instead, entirely outside this repo's own path.
WORKDIR = Path(tempfile.mkdtemp(prefix="trmcbo-core-build-"))

# Case-insensitive: catches "KOLON-BO", "KOLON", "Kolon", "kolon", inside
# hardcoded local-machine example paths a few src/ modules carry (e.g.
# "/workspace/KOLON-BO/data/..."). None of these are reachable from eval/'s
# import graph, but they still end up as string literals in the compiled
# binary if left as-is, so they're scrubbed in the staged copy before
# compiling -- src/ itself is never modified.
_COMPANY_NAME = re.compile(r"kolon", re.IGNORECASE)


def _scrub_company_name(stage: Path) -> None:
    for path in stage.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        scrubbed = _COMPANY_NAME.sub(lambda m: "TRMC" if m.group().isupper() else
                                     ("Trmc" if m.group()[0].isupper() else "trmc"), text)
        if scrubbed != text:
            path.write_text(scrubbed, encoding="utf-8")

COMPILER_DIRECTIVES = {
    "language_level": "3",
    "binding": True,
    "embedsignature": True,
    # Cython defaults to enforcing PEP 484 type annotations as runtime type
    # checks (`annotation_typing=True`), but plain Python never does -- a hint
    # like `def f(target: dict)` is purely decorative there. src/ has several
    # such annotations (e.g. bayesian_optimization.py's `_is_valid_range_target
    # (target: dict)`), and the enforced version rejects dict *subclasses*
    # (eval/scripts/audit_leakage.py's `_FieldSpy(dict)` diagnostic wrapper,
    # among others) that plain Python's duck typing accepts. Disabling this
    # keeps the compiled module's runtime behavior identical to source.
    "annotation_typing": False,
}


def discover_modules() -> list[str]:
    """Dotted module names for every `.py` file under `src/`, `__init__` excluded."""
    modules = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).with_suffix("")
        if rel.name == "__init__":
            continue
        modules.append(".".join(rel.parts))
    return modules


def package_dirs() -> list[Path]:
    """Every directory under `src/` that should exist as a package in `core/`."""
    dirs = {SRC}
    for path in SRC.rglob("*.py"):
        dirs.add(path.parent)
    return sorted(dirs)


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"expected CPython 3.10 (matches .venv-eval's ABI), got {sys.version}. "
            "Run this with .venv-build/bin/python."
        )

    if CORE.exists():
        shutil.rmtree(CORE)
    CORE.mkdir(parents=True)

    # Stage a copy of src/ to compile from, so build artifacts (.c, .so-in-place)
    # never land inside the real src/ tree, and the company name never reaches
    # the compiled output (path and file contents both scrubbed here).
    stage = WORKDIR / "src"
    shutil.copytree(SRC, stage)
    _scrub_company_name(stage)

    # Package markers: an (empty) __init__.py in every directory under core/ that
    # has one in src/, or a fresh empty one where src/ has none (src/ itself is
    # imported as a flat PYTHONPATH root, not a package, so no top-level __init__
    # is added there -- matching how `PYTHONPATH="src:."` works today).
    for d in package_dirs():
        rel = d.relative_to(SRC)
        target_dir = CORE / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        if rel != Path("."):
            (target_dir / "__init__.py").touch()

    modules = discover_modules()
    print(f"cythonizing {len(modules)} modules from {SRC} -> {CORE}")

    extensions = []
    for mod in modules:
        rel_path = Path(*mod.split(".")).with_suffix(".py")
        extensions.append(
            Extension(mod, [str(stage / rel_path)])
        )

    ext_modules = cythonize(
        extensions,
        compiler_directives=COMPILER_DIRECTIVES,
        build_dir=str(WORKDIR / "build"),
        language_level=3,
    )

    dist = Distribution({"name": "trmcbo_core", "ext_modules": ext_modules})
    dist.script_args = ["build_ext", "--inplace"]
    dist.parse_command_line()
    # --inplace resolves each extension's output path relative to the cwd's
    # package directories (which is why the dotted Extension names must match
    # stage/'s directory layout) -- so build from inside the staged copy, then
    # move the resulting .so files (and only those) into reproduce/core/.
    old_cwd = Path.cwd()
    try:
        os.chdir(stage)
        dist.run_command("build_ext")
    finally:
        os.chdir(old_cwd)

    # distutils' non-inplace build also drops copies under stage/build/lib.*/
    # before copying them to their inplace (correct) location -- skip those so
    # only the real, package-layout-mirroring .so files get moved.
    non_inplace_build = stage / "build"
    if non_inplace_build.exists():
        shutil.rmtree(non_inplace_build)

    so_count = 0
    for so_path in stage.rglob("*.so"):
        rel = so_path.relative_to(stage)
        target = CORE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(so_path), str(target))
        so_count += 1

    print(f"wrote {so_count} .so files under {CORE}")
    shutil.rmtree(WORKDIR)


if __name__ == "__main__":
    main()

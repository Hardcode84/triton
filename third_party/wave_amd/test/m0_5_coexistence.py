#!/usr/bin/env python3
"""M0.5 in-process coexistence gate for the wave_amd backend.

Proves that triton's `_C` MLIR/LLVM runtime and wave-mlir's `_mlir` runtime can
both live -- and build IR -- in a single Python process. This is the
precondition for the M2 converter, which reads finalized TTGIR via triton's
bindings and builds Wave IR via wave's bindings in the same process, passing
only plain data (bases, shapes, dtypes) across the boundary.

The two extensions are independently statically linked against different LLVM
pins with hidden visibility, so neither exports `llvm::`/`mlir::` symbols; this
test is the empirical confirmation that nothing clashes at load/use time.

Import order matches the real backend: triton is imported and exercised first
(it is always imported before the converter), then wave's package is added to
sys.path and exercised.

NB: this validates *IR construction* coexistence only. Raw HIP device calls are
NOT exercised here -- importing triton's `_C` breaks a separately-CDLL'd HIP
runtime, so device launches happen either in a subprocess (M0) or via triton's
own HIP driver (M3), never via a parallel raw CDLL.

Usage: python third_party/wave_amd/test/m0_5_coexistence.py
Env:   WAVE_REPO   wave-mlir repo root (default: /home/vano/7/7)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def wave_pkg() -> Path:
    root = Path(os.environ.get("WAVE_REPO", "/home/vano/7/7"))
    pkg = root / "build" / "python_packages" / "wave_mlir"
    if not (pkg / "mlir" / "dialects").is_dir():
        raise SystemExit(f"wave package not found at {pkg}; set WAVE_REPO")
    return pkg


def exercise_triton():
    """Create and use triton's MLIR context (keep refs alive)."""
    from triton._C.libtriton import ir, llvm

    llvm.init_targets()
    ctx = ir.context()
    ir.load_dialects(ctx)
    builder = ir.builder(ctx)
    module = builder.create_module()
    if module is None:
        raise SystemExit("triton create_module returned None")
    return ctx, module


def exercise_wave():
    """Create and use wave's MLIR context + dialects + passes (keep refs alive)."""
    sys.path.insert(0, str(wave_pkg()))
    # Importing wave_dsl loads wave's _mlir extension and calls register_passes().
    from mlir.dialects.wave_dsl import ModuleBuilder

    bld = ModuleBuilder()
    with bld:
        with bld.function("coexist_probe", inputs=[], kernel=True):
            pass
    module = bld.module
    if not module.operation.verify():
        raise SystemExit("wave module failed to verify")
    return bld, module


def main() -> int:
    print("[M0.5] exercising triton _C (context + dialects + module) ...")
    t_ctx, t_mod = exercise_triton()
    print("[M0.5]   triton MLIR context + module OK")

    print("[M0.5] exercising wave _mlir in the SAME process ...")
    w_bld, w_mod = exercise_wave()
    print("[M0.5]   wave context + register_dialects + register_passes + verify OK")

    # Both runtimes must still be alive and usable simultaneously.
    assert t_mod is not None and w_mod is not None
    # Touch both modules again after the other was created.
    _ = t_mod.str() if hasattr(t_mod, "str") else str(t_mod)
    _ = w_mod.operation.verify()
    print("[M0.5] PASS: triton _C and wave _mlir coexist and build IR in one process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

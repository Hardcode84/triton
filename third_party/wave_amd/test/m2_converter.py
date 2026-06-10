#!/usr/bin/env python3
"""[M2] structural finalized-TTGIR -> Wave-dialect converter smoke test.

Compiles the 1-D masked-copy kernel to FINALIZED TTGIR through triton's real
AMD pipeline (the exact ``make_ttir`` / ``make_ttgir`` the wave_amd backend
inherits from HIPBackend, identical to m1b), converts it to a Wave-dialect
module with the M2 structural converter, asserts the result verifies, and
checks that it structurally mirrors the kernel (lane_id / workgroup_id / cmpi /
where / ptr_add / load / store). All in one process; no GPU, no asm.

Usage: python third_party/wave_amd/test/m2_converter.py
Env:   WAVE_REPO   wave-mlir repo root (default: /home/vano/7/7)
"""

from __future__ import annotations

import sys
from pathlib import Path

BLOCK = 32
NUM_WARPS = 1
ARCH = "gfx1100"
WARP_SIZE = 32  # gfx11 wavefront

# Import the converter that lives in the sibling backend/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def build_finalized_ttgir():
    """Compile masked-copy to finalized TTGIR; return (module, context).

    The context is returned so the caller keeps it alive for the module's life.
    """
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd.compiler import HIPBackend
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir

    @triton.jit
    def masked_copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, v, mask=mask)

    target = GPUTarget("hip", ARCH, WARP_SIZE)
    backend = HIPBackend(target)
    options = backend.parse_options({"num_warps": NUM_WARPS})

    src = ASTSource(
        fn=masked_copy,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "n": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": BLOCK},
    )

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()

    metadata = {}
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    return module, context


def main() -> int:
    from wave_converter import convert, wave_ir_text

    print(f"[M2] compiling masked-copy -> finalized TTGIR ({ARCH}, {NUM_WARPS} warp, BLOCK={BLOCK}) ...")
    ttgir_module, _context = build_finalized_ttgir()
    if "ttg." not in ttgir_module.str():
        raise SystemExit("expected finalized TTGIR (no ttg. ops found)")

    print("[M2] converting finalized TTGIR -> Wave dialect ...")
    bld = convert(ttgir_module, ARCH)
    text = wave_ir_text(bld)

    if not bld.module.operation.verify():
        sys.stderr.write(text + "\n")
        raise SystemExit("[M2] FAIL: produced Wave module did not verify")

    required = [
        "func.func @masked_copy",
        "wave.kernel",
        "waveamdmachine.target",
        "wave.lane_id",
        "wave.workgroup_id",
        "wave.cmpi",
        "wave.where",
        "wave.ptr_add",
        "wave.load",
        "wave.store",
    ]
    missing = [tok for tok in required if tok not in text]
    if missing:
        sys.stderr.write(text + "\n")
        raise SystemExit(f"[M2] FAIL: produced Wave IR missing tokens: {missing}")

    print(text)
    print(f"[M2] PASS: structural TTGIR -> Wave conversion verified ({len(required)} markers present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

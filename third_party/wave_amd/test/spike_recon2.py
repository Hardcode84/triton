#!/usr/bin/env python3
"""[generalize recon] inspect address DAG + layouts for representative kernels.

Goal: validate the address-synthesis design for the generalized converter --
can per-dim strides be extracted from the addptr DAG (constant AND runtime), and
what do the access layouts look like for 1-D masked copy vs 2-D strided tiles.
"""

from __future__ import annotations

import sys

ARCH = "gfx1100"
WARP_SIZE = 32


def compile_ttgir(fn, signature, constexprs, num_warps):
    import triton  # noqa: F401
    from triton.backends.compiler import GPUTarget
    from triton.backends.amd.compiler import HIPBackend
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir

    target = GPUTarget("hip", ARCH, WARP_SIZE)
    backend = HIPBackend(target)
    options = backend.parse_options({"num_warps": num_warps})
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    md = {}
    m = src.make_ir(target, options, codegen_fns, module_map, context)
    m = backend.make_ttir(m, md, options)
    m = backend.make_ttgir(m, md, options)
    return m, context


def describe(ll):
    if ll is None:
        return "<none>"
    b = dict(ll.bases)
    return "reg={} lane={} warp={} block={}".format(b.get("register", []), b.get("lane", []), b.get("warp", []),
                                                    b.get("block", []))


def report(name, fn, signature, constexprs, num_warps):
    print("=" * 78)
    print(f"[{name}] num_warps={num_warps}")
    print("=" * 78)
    m, _ctx = compile_ttgir(fn, signature, constexprs, num_warps)
    ops = []
    m.walk(ops.append)
    for op in ops:
        if op.get_name() in ("tt.load", "tt.store"):
            print(
                f"  {op.get_name()}: access layout {describe(op.get_result(0).get_tensor_layout() if op.get_num_results() else op.get_operand(1 if op.get_name()=='tt.store' else 0).get_tensor_layout())}"
            )
    print("  ----- TTGIR -----")
    print(m.str())


def main() -> int:
    import triton
    import triton.language as tl

    @triton.jit
    def masked_copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, v, mask=mask)

    @triton.jit
    def strided_copy(x_ptr, y_ptr, stride_m, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        offs = rm[:, None] * stride_m + rn[None, :]
        v = tl.load(x_ptr + offs)
        tl.store(y_ptr + offs, v)

    report("masked_copy", masked_copy, {"x_ptr": "*fp32", "y_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
           {"BLOCK": 32}, 1)
    report("strided_copy 32x8 runtime-stride", strided_copy,
           {"x_ptr": "*fp32", "y_ptr": "*fp32", "stride_m": "i32", "BM": "constexpr", "BN": "constexpr"},
           {"BM": 32, "BN": 8}, 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

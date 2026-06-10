#!/usr/bin/env python3
"""[spike step 1] convert_layout recon.

Compile a few candidate kernels to FINALIZED TTGIR through the real AMD pipeline
(the same make_ttir/make_ttgir the wave_amd backend inherits), then for every
``ttg.convert_layout`` print the src/dst tensor types and their LinearLayout
bases. Goal: pick the simplest kernel that retains ONE affine cross-warp/
cross-lane convert_layout to drive the LDS-roundtrip spike.

Usage: PYTHONPATH=/home/vano/triton/python TRITON_BACKENDS_IN_TREE=1 \
       python third_party/wave_amd/test/spike_recon.py
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


def describe_layout(ll):
    if ll is None:
        return "  <no layout>"
    bases = dict(ll.bases)
    lines = []
    for axis in ("register", "lane", "warp", "block"):
        b = bases.get(axis, [])
        lines.append(f"    {axis:9s} (n={len(b)}): {b}")
    try:
        outs = ll.out_dims
    except Exception:
        outs = "?"
    return f"  out_dims={outs}\n" + "\n".join(lines)


def report(name, fn, signature, constexprs, num_warps, dump_ttgir=False):
    print("=" * 78)
    print(f"[{name}] num_warps={num_warps} sig={signature} constexprs={constexprs}")
    print("=" * 78)
    try:
        m, _ctx = compile_ttgir(fn, signature, constexprs, num_warps)
    except Exception as exc:  # noqa: BLE001
        print(f"  COMPILE FAILED: {type(exc).__name__}: {exc}")
        return
    text = m.str()
    ops = []
    m.walk(ops.append)
    cvts = [op for op in ops if op.get_name() == "ttg.convert_layout"]
    print(f"  ops={len(ops)}  convert_layout count={len(cvts)}")
    hist = {}
    for op in ops:
        hist[op.get_name()] = hist.get(op.get_name(), 0) + 1
    print("  op histogram:", {k: v for k, v in sorted(hist.items())})
    for i, op in enumerate(cvts):
        src = op.get_operand(0)
        dst = op.get_result(0)
        sl = src.get_tensor_layout()
        dl = dst.get_tensor_layout()
        nreg = len(dict(sl.bases).get("register", [])) if sl else "?"
        print(f"\n  --- convert_layout #{i}  (src regs/lane = 2^{nreg}) ---")
        print(f"  src type: {src.get_type().str() if hasattr(src.get_type(), 'str') else src.get_type()}")
        print("  src layout:")
        print(describe_layout(sl))
        print("  dst layout:")
        print(describe_layout(dl))
    if dump_ttgir:
        print("\n  ----- TTGIR -----")
        print(text)


def main() -> int:
    import triton
    import triton.language as tl

    @triton.jit
    def transpose(x_ptr, y_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        x = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        xt = tl.trans(x)
        tl.store(y_ptr + rn[:, None] * BM + rm[None, :], xt)

    @triton.jit
    def rowsum(x_ptr, y_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        x = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        s = tl.sum(x, axis=1)
        tl.store(y_ptr + rm, s)

    tsig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "BM": "constexpr", "BN": "constexpr"}

    report("transpose 32x2 w1", transpose, tsig, {"BM": 32, "BN": 2}, 1)
    report("transpose 2x32 w1", transpose, tsig, {"BM": 2, "BN": 32}, 1)
    report("transpose 32x4 w1", transpose, tsig, {"BM": 32, "BN": 4}, 1)
    report("transpose 32x32 w1", transpose, tsig, {"BM": 32, "BN": 32}, 1, dump_ttgir=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

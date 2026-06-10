#!/usr/bin/env python3
"""M1b: validate the finalized-TTGIR layout + dtype bindings the converter needs.

The M2 converter walks finalized TTGIR through triton's ``_C`` bindings and
relies on two additions to triton's Python IR surface (built into ``libtriton``
in this milestone):

  * ``Value.get_tensor_layout()`` -> ``LinearLayout`` for any tensor value,
    exposing the distributed (register/lane/warp/block) layout assigned by the
    AMD layout pipeline as plain basis data. Returns ``None`` for non-tensor
    values or tensors without an encoding.
  * structural dtype introspection on ``Type`` -- ``is_ranked_tensor``,
    ``get_element_type``, ``is_pointer_type``, ``get_pointee_type``,
    ``is_floating``, ``get_bitwidth`` -- each looking through a ranked tensor to
    its element type so it can be called on scalar or tensor types alike.

It compiles a 1-D masked-copy kernel through triton's *real* AMD pipeline (the
exact ``make_ttir`` / ``make_ttgir`` the wave_amd backend inherits from
HIPBackend) to FINALIZED TTGIR, then walks it and asserts the new bindings
return well-formed layout/dtype data. No GPU is required (no launch, no asm).

Usage: python third_party/wave_amd/test/m1b_layout_bindings.py
"""

from __future__ import annotations

BLOCK = 1024
NUM_WARPS = 4
ARCH = "gfx1100"
WARP_SIZE = 32  # gfx11 wavefront
HW_AXES = {"register", "lane", "warp", "block"}


def build_finalized_ttgir():
    """Compile a masked-copy kernel to finalized TTGIR; return (module, context).

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


def collect_ops(module):
    ops = []
    module.walk(lambda op: ops.append(op))
    return ops


def check_layouts(ops):
    """Every tensor result must expose a HW-axis LinearLayout; return them."""
    layouts = []
    for op in ops:
        for i in range(op.get_num_results()):
            value = op.get_result(i)
            ll = value.get_tensor_layout()
            if ll is None:
                continue
            in_dims = ll.get_in_dim_names()
            out_dims = ll.get_out_dim_names()
            layouts.append((op.get_name(), value.get_shape(), in_dims, out_dims, ll))

    if not layouts:
        raise SystemExit("no tensor value exposed a LinearLayout via get_tensor_layout()")

    for name, shape, in_dims, out_dims, ll in layouts:
        extra = set(in_dims) - HW_AXES
        if extra:
            raise SystemExit(f"{name}: layout has non-HW in dims {sorted(extra)}")
        for need in ("register", "lane", "warp"):
            if need not in in_dims:
                raise SystemExit(f"{name}: layout missing '{need}' axis: {in_dims}")
        if len(out_dims) != 1:
            raise SystemExit(f"{name} shape={shape}: expected rank-1 out dims, got {out_dims}")
        # The layout must round-trip through plain basis data.
        bases = dict(ll.bases)
        if set(bases) != set(in_dims):
            raise SystemExit(f"{name}: bases keys {sorted(bases)} != in dims {sorted(in_dims)}")
    return layouts


def check_load_layout(ops):
    """The loaded [BLOCK] f32 tensor: lane spans the wavefront, warp the warps."""
    loads = [op for op in ops if op.get_name() == "tt.load"]
    if not loads:
        raise SystemExit("no tt.load in finalized TTGIR")
    value = loads[0].get_result(0)
    if value.get_shape() != [BLOCK]:
        raise SystemExit(f"tt.load result shape {value.get_shape()} != [{BLOCK}]")
    ll = value.get_tensor_layout()
    if ll is None:
        raise SystemExit("tt.load result has no layout")
    bases = dict(ll.bases)
    n_lane = len(bases["lane"])
    n_warp = len(bases["warp"])
    n_reg = len(bases["register"])
    if 2**n_lane != WARP_SIZE:
        raise SystemExit(f"lane spans 2**{n_lane} != {WARP_SIZE}")
    if 2**n_warp != NUM_WARPS:
        raise SystemExit(f"warp spans 2**{n_warp} != {NUM_WARPS}")
    # register x lane x warp must cover all BLOCK elements (single block).
    covered = (2**n_reg) * WARP_SIZE * NUM_WARPS
    if covered != BLOCK:
        raise SystemExit(f"register*lane*warp = {covered} != {BLOCK}")
    return loads[0], value, ll


def check_dtypes(load_op, load_value):
    """Exercise the look-through dtype helpers on the load result + ptr operand."""
    rty = load_value.get_type()
    if not rty.is_ranked_tensor():
        raise SystemExit("tt.load result is not a ranked tensor")
    if rty.is_pointer_type():
        raise SystemExit("tt.load result wrongly classified as pointer")
    if not rty.is_floating():
        raise SystemExit("tt.load element is not floating")
    if rty.get_bitwidth() != 32:
        raise SystemExit(f"tt.load element bitwidth {rty.get_bitwidth()} != 32")
    elem = rty.get_element_type()
    if elem.is_ranked_tensor():
        raise SystemExit("element type of a tensor must not itself be a tensor")
    if elem.get_bitwidth() != 32 or not elem.is_floating():
        raise SystemExit("element type is not f32")

    # operand 0 of tt.load is a tensor of !tt.ptr<f32>.
    ptr_operand = load_op.get_operand(0)
    pty = ptr_operand.get_type()
    if not pty.is_ranked_tensor():
        raise SystemExit("tt.load addr operand is not a tensor (of pointers)")
    if not pty.is_pointer_type():
        raise SystemExit("tt.load addr operand not classified as pointer")
    pointee = pty.get_pointee_type()
    if pointee is None:
        raise SystemExit("pointer operand has no pointee type")
    if not pointee.is_floating() or pointee.get_bitwidth() != 32:
        raise SystemExit("tt.load addr pointee is not f32")
    if pointee.is_pointer_type():
        raise SystemExit("f32 pointee wrongly classified as pointer")


def main() -> int:
    print(f"[M1b] compiling masked-copy -> finalized TTGIR ({ARCH}, {NUM_WARPS} warps, BLOCK={BLOCK}) ...")
    module, _context = build_finalized_ttgir()
    text = module.str()
    if "ttg." not in text:
        raise SystemExit("expected finalized TTGIR (no ttg. ops found)")
    ops = collect_ops(module)
    print(f"[M1b]   walked {len(ops)} ops")

    layouts = check_layouts(ops)
    print(f"[M1b]   get_tensor_layout() returned LinearLayouts for {len(layouts)} tensor results")

    load_op, load_value, ll = check_load_layout(ops)
    bases = dict(ll.bases)
    spans = {axis: 2**len(bases[axis]) for axis in ("register", "lane", "warp", "block")}
    print(f"[M1b]   tt.load [{BLOCK}]xf32 layout spans (elements per axis): {spans}")
    print(f"[M1b]   in dims={ll.get_in_dim_names()} out dims={ll.get_out_dim_names()}")

    check_dtypes(load_op, load_value)
    print("[M1b]   dtype introspection (ranked-tensor look-through) OK on result + ptr operand")

    print("[M1b] PASS: layout + dtype bindings expose finalized-TTGIR data for the converter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

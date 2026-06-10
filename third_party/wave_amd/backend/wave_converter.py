"""wave_amd: structural finalized-TTGIR -> Wave-dialect converter.

Builds, in one Python process and with no GPU or assembly, a Wave-dialect MLIR
module that structurally mirrors a *finalized* TTGIR kernel. It reads the TTGIR
through triton's ``_C`` bindings (the op walk plus typed value/layout/dtype
introspection) and emits Wave IR through the wave-mlir Python DSL
(``mlir.dialects.wave_dsl``), passing only plain data (op names, dtypes, layout
*bases*, attribute ints) across the two runtimes' boundary.

Value model -- ``RegVal``:

  A distributed TTGIR tensor value maps to a ``RegVal``: the list of per-register
  ``!wave.simd<T, W>`` values (one *slot* per element a lane holds) together with
  its ``LinearLayout``. Slot ``r`` at lane ``l`` (warp ``w`` at runtime) holds the
  tensor element at coordinate ``XOR`` of the selected register/lane/warp basis
  vectors -- always read from ``.bases`` (never ``.apply``, which rebuilds query
  StringAttrs from a singleton context that differs from the layout's and aborts).
  ``n_reg = 2 ** len(register bases)``; single-element-per-lane tensors are a
  one-slot ``RegVal``. Uniform scalars and *base pointers* stay raw Wave values
  (a base-pointer splat is fused into the following ``ptr_add`` by aliasing).

Op classes handled (single CTA; affine layouts):

  * Elementwise / index arithmetic, ``make_range`` (iota synthesized per-slot
    from its slice-layout bases), ``expand_dims`` / ``broadcast`` / ``trans``
    (re-slotting by matching register-base coordinates), ``addptr`` / ``load`` /
    ``store`` (per slot).
  * ``ttg.convert_layout``: an LDS store -> ``wave.barrier`` -> load roundtrip,
    synthesized from the src/dst ``LinearLayout`` deltas with a naive row-major
    shared layout (correct, not bank-optimal). Cross-warp deltas thread the
    workgroup ``wave.subgroup_id`` into the addresses.
  * The elementwise mask is hoisted into one ``wave.where(mask)`` region.

Out of scope (raise ``NotImplementedError``): non-affine / XOR-swizzled layouts
(dot operands, bank-optimal shared), atomics, reductions, multi-CTA (``block``
bases).

Public API:
  * ``convert(ttgir_module, arch) -> ModuleBuilder``
  * ``wave_ir_text(bld) -> str``

The Wave DSL is imported lazily inside :func:`convert` (after the wave package
is placed on ``sys.path``) so the triton-first import order required for
in-process coexistence is preserved; there are no top-level Wave imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# arith::CmpIPredicate enum value -> wave.cmpi predicate keyword.
# wave's AMDGPU lowering (WaveAMDMachine getU32CmpKind) only supports the
# unsigned U32 comparisons {eq, ne, ult, ule, ugt, uge}. triton emits *signed*
# index comparisons (e.g. `offs < n` -> slt), so the signed predicates are
# folded onto their unsigned equivalents. This is sound for the elementwise
# masked-copy scope, whose compared values (program_id*BLOCK + iota, and the
# bound n) are non-negative.
_CMPI_PRED = {
    0: "eq",
    1: "ne",
    2: "ult",  # slt
    3: "ule",  # sle
    4: "ugt",  # sgt
    5: "uge",  # sge
    6: "ult",
    7: "ule",
    8: "ugt",
    9: "uge",
}

# Structural ops the walk yields that the body translation never emits.
_SKIP_OPS = ("builtin.module", "tt.func", "tt.return")

_DEFAULT_WAVE_REPO = "/home/vano/7/7"


def _wave_pkg_path() -> str:
    root = Path(os.environ.get("WAVE_REPO", _DEFAULT_WAVE_REPO))
    return str(root / "build" / "python_packages" / "wave_mlir")


def _import_dsl():
    """Place the wave package on ``sys.path`` (honoring ``WAVE_REPO``) and import
    the Wave DSL lazily, keeping triton's ``_C`` imported first."""
    pkg = _wave_pkg_path()
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from mlir.dialects import wave_dsl as dsl
    return dsl


def _wave_translate_bin() -> str:
    env = os.environ.get("WAVE_TRANSLATE")
    if env:
        return env
    root = os.environ.get("WAVE_REPO", _DEFAULT_WAVE_REPO)
    return str(Path(root) / "build" / "bin" / "wave-translate")


def wave_to_amdgpu_asm(module_text: str) -> str:
    """Lower high-level Wave IR text to AMDGPU asm via ``wave-translate``.

    The target arch travels in the module's ``waveamdmachine.target`` attribute
    (set by :func:`convert`), so no arch flag is needed here. wave-translate runs
    in a subprocess against wave-mlir's own statically-linked LLVM; only text
    crosses the boundary, so it never clashes with triton's ``_C`` LLVM in-process.
    """
    tool = _wave_translate_bin()
    if not Path(tool).exists():
        raise RuntimeError(f"wave_amd: wave-translate not found at {tool}; set WAVE_TRANSLATE")
    proc = subprocess.run([tool, "--wave-to-amdgpu-asm", "-"], input=module_text, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wave_amd: wave-translate failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def _wave_scalar_type(ty, dsl):
    """Map a TTGIR scalar (or tensor element) type to its Wave scalar type.

    The triton dtype helpers look through a ranked tensor to its element type,
    so this is well defined on scalar function args and tensor results alike.
    """
    if ty.is_pointer_type():
        pointee = ty.get_pointee_type()
        return dsl.ptr_type(_wave_scalar_type(pointee, dsl), dsl.global_address_space())
    if ty.is_floating():
        bw = ty.get_bitwidth()
        if bw == 16:
            return dsl.f16()
        if bw == 32:
            return dsl.f32()
        raise NotImplementedError(f"wave_amd M2: unsupported float width {bw}")
    bw = ty.get_bitwidth()
    int_types = {1: dsl.i1, 8: dsl.i8, 32: dsl.i32, 64: dsl.i64}
    if bw not in int_types:
        raise NotImplementedError(f"wave_amd M2: unsupported int width {bw}")
    return int_types[bw]()


# ---------------------------------------------------------------------------
# Value model
#
# A vmap entry is one of:
#   * a raw Wave ``Value``            -- a uniform scalar (program id, n, a
#                                        constant, a runtime stride).
#   * a ``Lin``                       -- a distributed *index* tensor, kept
#                                        symbolic as an affine function of the
#                                        tensor's coordinates (so broadcast /
#                                        expand_dims / runtime strides are pure
#                                        bookkeeping; SIMD is synthesized only at
#                                        the consuming cmpi / load / store).
#   * a ``PtrVal``                    -- a distributed pointer: a uniform base
#                                        plus a ``Lin`` byte/elt offset.
#   * a ``RegVal``                    -- a distributed *data* tensor: the actual
#                                        per-register SIMD slots + its layout.
# ---------------------------------------------------------------------------


class RegVal:
    """A distributed TTGIR data tensor: per-register SIMD slots + its layout.

    ``slots[r]`` is the ``!wave.simd<T, W>`` holding register slot ``r`` (the
    r-th element each lane owns); ``layout`` is the tensor's ``LinearLayout``.
    """

    __slots__ = ("slots", "layout")

    def __init__(self, slots, layout):
        self.slots = list(slots)
        self.layout = layout


class Lin:
    """A distributed index tensor as an affine function of tensor coordinates.

    ``value(coord) = const + sum_dim coeffs[dim] * coord[dim]``. ``coeffs`` maps
    a tensor dimension index to a stride (a Python ``int`` or a uniform Wave
    ``Value`` for runtime strides); ``const`` is an ``int``, a uniform ``Value``,
    or ``None`` (zero). The actual per-lane SIMD is materialized by
    :func:`_synth_lin` against the *consuming* op's layout.
    """

    __slots__ = ("coeffs", "const")

    def __init__(self, coeffs, const):
        self.coeffs = dict(coeffs)
        self.const = const


class PtrVal:
    """A distributed pointer: a uniform base Wave value + a ``Lin`` offset."""

    __slots__ = ("base", "lin")

    def __init__(self, base, lin):
        self.base = base
        self.lin = lin


def _n_reg(layout):
    """Number of register slots a value with ``layout`` holds per lane."""
    if layout is None:
        return 1
    return 2**len(dict(layout.bases).get("register", []))


def _slot(value, r):
    """Slot ``r`` of a ``RegVal``, or a uniform value broadcast to every slot."""
    return value.slots[r] if isinstance(value, RegVal) else value


# --- symbolic affine arithmetic on Lin ------------------------------------


def _const_add(x, y):
    if x is None:
        return y
    if y is None:
        return x
    if isinstance(x, int) and isinstance(y, int):
        return x + y
    raise NotImplementedError("wave_amd: runtime + runtime constant index add unsupported")


def _scalar_mul(s, k):
    if isinstance(s, int) and isinstance(k, int):
        return s * k
    if isinstance(s, int) and s == 1:
        return k
    if isinstance(k, int) and k == 1:
        return s
    raise NotImplementedError("wave_amd: runtime * runtime stride unsupported")


def _lin_add(a, b):
    coeffs = dict(a.coeffs)
    for d, s in b.coeffs.items():
        if d in coeffs:
            raise NotImplementedError("wave_amd: overlapping affine index dimension")
        coeffs[d] = s
    return Lin(coeffs, _const_add(a.const, b.const))


def _lin_scale(lin, k):
    return Lin({d: _scalar_mul(s, k)
                for d, s in lin.coeffs.items()}, None if lin.const is None else _scalar_mul(lin.const, k))


def _lin_mul(a, b):
    if not a.coeffs:
        return _lin_scale(b, a.const if a.const is not None else 0)
    if not b.coeffs:
        return _lin_scale(a, b.const if b.const is not None else 0)
    raise NotImplementedError("wave_amd: non-affine index (coeff * coeff)")


def _lin_shift(lin, axis):
    """expand_dims: insert a unit axis at ``axis`` (shift higher dims up)."""
    return Lin({(d if d < axis else d + 1): s for d, s in lin.coeffs.items()}, lin.const)


# --- SIMD synthesis from a layout's bases ----------------------------------


def _reg_coord(reg_bases, r, dim):
    """The ``dim`` component of register slot ``r``'s coordinate (XOR of bases)."""
    v = 0
    for k, base in enumerate(reg_bases):
        if (r >> k) & 1:
            v ^= base[dim]
    return v


def _splat_int(fb, dsl, v, width):
    return fb.splat(fb.constant(dsl.i32(), v), dsl.i32(), width=width)


def _as_simd(fb, dsl, v, width):
    """Broadcast an ``int`` or uniform Wave value to a ``simd<i32, W>``."""
    if isinstance(v, int):
        return _splat_int(fb, dsl, v, width)
    return fb.splat(v, dsl.i32(), width=width)


def _mul_coeff(fb, dsl, coord, coeff, width):
    if isinstance(coeff, int):
        return coord if coeff == 1 else fb.muli(coord, _splat_int(fb, dsl, coeff, width))
    return fb.muli(coord, fb.splat(coeff, dsl.i32(), width=width))


def _bit_poly(fb, dsl, x, deltas, width):
    """``sum_k bit_k(x) * deltas[k]`` as a ``simd<i32, W>`` (or None if all 0).

    Fast path when ``deltas[k] == c * 2**k`` for all k (the contiguous case):
    just ``c * x``. Otherwise emit a per-bit shift/mask/mul/add chain, which
    covers bit-interleaved lane layouts (e.g. ``(lane >> 3) & 3``).
    """
    if all(d == 0 for d in deltas):
        return None
    c, clean = None, True
    for k, d in enumerate(deltas):
        if d % (1 << k) != 0:
            clean = False
            break
        q = d >> k
        if c is None:
            c = q
        elif c != q:
            clean = False
            break
    if clean:
        return x if c == 1 else fb.muli(x, _splat_int(fb, dsl, c, width))
    bk = dsl.BinaryKind
    acc = None
    for k, d in enumerate(deltas):
        if d == 0:
            continue
        sh = fb.binary(bk.ShRUI, x, _splat_int(fb, dsl, k, width)) if k else x
        bit = fb.binary(bk.AndI, sh, _splat_int(fb, dsl, 1, width))
        term = bit if d == 1 else fb.muli(bit, _splat_int(fb, dsl, d, width))
        acc = term if acc is None else fb.addi(acc, term)
    return acc


def _synth_lin(fb, dsl, lin, layout, width, warp_id=None):
    """Materialize a ``Lin`` against ``layout`` as a ``RegVal`` of i32 offsets.

    Slot ``r``'s value at lane ``l`` (warp ``w``) is ``const + sum_dim coeff[dim]
    * coord_dim(r, l, w)``, where ``coord_dim`` is the register (XOR of register
    bases) plus the lane and warp contributions (``_bit_poly`` over the lane /
    warp bases). The lane/warp-varying part is shared across slots; only the
    register constant differs. ``warp_id`` is the per-lane warp id SIMD (required
    only when the layout has warp bases).
    """
    bases = dict(layout.bases)
    reg_bases = bases.get("register", [])
    lane_bases = bases.get("lane", [])
    warp_bases = bases.get("warp", [])
    if warp_bases and warp_id is None:
        raise NotImplementedError("wave_amd: warp-distributed layout but no warp id available")
    n_reg = 2**len(reg_bases)
    lane = fb.lane_id(dsl.i32(), width=width)

    def coord_var(dim):
        lv = _bit_poly(fb, dsl, lane, [b[dim] for b in lane_bases], width)
        if warp_bases:
            wv = _bit_poly(fb, dsl, warp_id, [b[dim] for b in warp_bases], width)
            if wv is not None:
                lv = wv if lv is None else fb.addi(lv, wv)
        return lv

    lanevar = {dim: coord_var(dim) for dim in lin.coeffs}
    slots = []
    for r in range(n_reg):
        acc = None
        for dim, coeff in lin.coeffs.items():
            coord = lanevar[dim]
            cpart = _reg_coord(reg_bases, r, dim)
            if cpart != 0:
                cs = _splat_int(fb, dsl, cpart, width)
                coord = cs if coord is None else fb.addi(coord, cs)
            if coord is None:
                continue
            term = _mul_coeff(fb, dsl, coord, coeff, width)
            acc = term if acc is None else fb.addi(acc, term)
        if lin.const is not None and not (isinstance(lin.const, int) and lin.const == 0):
            cs = _as_simd(fb, dsl, lin.const, width)
            acc = cs if acc is None else fb.addi(acc, cs)
        slots.append(acc if acc is not None else _splat_int(fb, dsl, 0, width))
    return RegVal(slots, layout)


def _synth_addr(fb, dsl, ptrval, layout, width, warp_id=None):
    """Materialize a ``PtrVal`` against ``layout`` as a ``RegVal`` of pointers."""
    offs = _synth_lin(fb, dsl, ptrval.lin, layout, width, warp_id)
    return RegVal([fb.ptr_add(ptrval.base, s) for s in offs.slots], layout)


def _rowmajor_lin(sizes):
    """A ``Lin`` whose coeffs are the row-major strides of a tile of ``sizes``.

    Used as the *shared* (LDS) linearization for a convert_layout: a naive,
    unswizzled row-major arena (correct, not bank-optimal).
    """
    coeffs, stride = {}, 1
    for d in range(len(sizes) - 1, -1, -1):
        coeffs[d] = stride
        stride *= sizes[d]
    return Lin(coeffs, None)


def _convert_layout(fb, dsl, src, src_ll, dst_ll, elem, simd, width, warp_id=None):
    """Realize a ``ttg.convert_layout`` as an LDS store -> barrier -> load.

    Store each src slot into a naive row-major LDS arena at its src-distribution
    offset, barrier so every store is visible, then load each dst slot from its
    dst-distribution offset (sequenced ``after`` the barrier). The shared offset
    is the same row-major flatten for both sides, so the data lands where the dst
    distribution expects it. With warp bases the offsets carry the ``warp_id``,
    so the arena spans the whole workgroup tile and the (workgroup-wide)
    ``s_barrier`` from the token chain is the cross-warp fence. Tokens thread the
    store -> barrier -> load order (-> the load-bearing ``s_barrier`` +
    ``lgkmcnt`` fence).
    """
    rm = _rowmajor_lin([size for _name, size in src_ll.out_dims])
    lds = fb.lds_base(elem)
    src_off = _synth_lin(fb, dsl, rm, src_ll, width, warp_id)
    dst_off = _synth_lin(fb, dsl, rm, dst_ll, width, warp_id)
    store_toks = [fb.store(src.slots[r], fb.ptr_add(lds, src_off.slots[r])) for r in range(len(src_off.slots))]
    bar = fb.barrier(*store_toks)
    out = [fb.load(fb.ptr_add(lds, dst_off.slots[r]), simd, after=bar)[0] for r in range(len(dst_off.slots))]
    return RegVal(out, dst_ll)


def _lds_bytes(ops):
    """Bytes of LDS the kernel needs: the largest convert_layout tile (or 0)."""
    total = 0
    for op in ops:
        if op.get_name() == "ttg.convert_layout":
            res = op.get_result(0)
            n = 1
            for _name, size in res.get_tensor_layout().out_dims:
                n *= size
            total = max(total, n * (res.get_type().get_bitwidth() // 8))
    return total


def _scope_gate(ops, threads_per_warp, allow_warp=False):
    """Gate to the supported layout class and return the wavefront width W.

    Allows ``register`` + ``lane`` bases (and ``warp`` when ``allow_warp``);
    rejects multi-CTA (``block``) bases. The wavefront width is the module
    threads-per-warp; per-tensor lane coverage may be narrower (broadcast).
    """
    for op in ops:
        for i in range(op.get_num_results()):
            ll = op.get_result(i).get_tensor_layout()
            if ll is None:
                continue
            bases = dict(ll.bases)
            if len(bases.get("block", [])) != 0:
                raise NotImplementedError("wave_amd: multi-CTA (block bases) unsupported")
            if not allow_warp and len(bases.get("warp", [])) != 0:
                raise NotImplementedError("wave_amd: multi-warp (warp bases) not yet supported")
    return threads_per_warp


def _first_mem_index(body):
    """Index of the first ``tt.load``/``tt.store`` in ``body``, or None."""
    for i, op in enumerate(body):
        if op.get_name() in ("tt.load", "tt.store"):
            return i
    return None


def _mask_operand_id(op):
    """Id of a masked memory op's mask operand, or None when unmasked.

    ``tt.load`` operands are ``[ptr]`` or ``[ptr, mask]``; ``tt.store`` operands
    are ``[ptr, val]`` or ``[ptr, val, mask]``.
    """
    name = op.get_name()
    n = op.get_num_operands()
    if name == "tt.load":
        return op.get_operand(1).id() if n >= 2 else None
    if name == "tt.store":
        return op.get_operand(2).id() if n >= 3 else None
    return None


def _uses_warp(ops):
    """True if any tensor result carries a warp-distributed layout."""
    for op in ops:
        for i in range(op.get_num_results()):
            ll = op.get_result(i).get_tensor_layout()
            if ll is not None and len(dict(ll.bases).get("warp", [])) != 0:
                return True
    return False


def _translate_op(op, fb, vmap, width, dsl, warp_id=None):
    """Translate one TTGIR body op into Wave IR, recording results in ``vmap``.

    Index/pointer ops fold into symbolic ``Lin`` / ``PtrVal`` (no IR emitted);
    SIMD is materialized only at ``cmpi`` / ``load`` / ``store``. Data ops emit
    ``RegVal`` slots. ``n_reg == 1`` reproduces the single-element-per-lane case.
    """
    name = op.get_name()

    def operand(i):
        return vmap[op.get_operand(i).id()]

    def result_layout():
        return op.get_result(0).get_tensor_layout()

    def bind(value):
        vmap[op.get_result(0).id()] = value

    if name == "arith.constant":
        if result_layout() is None:
            bind(fb.constant(_wave_scalar_type(op.get_result(0).get_type(), dsl), op.get_constant_value()))
        else:
            bind(Lin({}, op.get_constant_value()))
    elif name == "tt.get_program_id":
        bind(fb.workgroup_id(0))
    elif name in ("arith.muli", "arith.addi"):
        if result_layout() is None:
            bind((fb.muli if name == "arith.muli" else fb.addi)(operand(0), operand(1)))
        else:
            bind((_lin_mul if name == "arith.muli" else _lin_add)(operand(0), operand(1)))
    elif name == "arith.cmpi":
        lay = result_layout()
        if lay is None:
            raise NotImplementedError("wave_amd: scalar cmpi unsupported")
        pred = _CMPI_PRED[op.get_int_attr("predicate")]
        a = _synth_lin(fb, dsl, operand(0), lay, width, warp_id)
        b = _synth_lin(fb, dsl, operand(1), lay, width, warp_id)
        bind(RegVal([fb.cmpi(pred, a.slots[r], b.slots[r]) for r in range(_n_reg(lay))], lay))
    elif name == "tt.make_range":
        if op.get_int_attr("start") != 0:
            raise NotImplementedError("wave_amd: tt.make_range with start != 0 is unsupported")
        # iota along its (1-D) dimension; expand_dims places it in the final tile.
        bind(Lin({0: 1}, None))
    elif name == "tt.splat":
        result = op.get_result(0)
        src_ty = op.get_operand(0).get_type()
        if result.get_type().is_pointer_type():
            # Scalar base-pointer splat: fuse into the following ptr_add via a
            # symbolic PtrVal (no op emitted).
            bind(PtrVal(operand(0), Lin({}, None)))
        elif src_ty.is_floating():
            elem = _wave_scalar_type(src_ty, dsl)
            sp = fb.splat(operand(0), elem, width=width)
            bind(RegVal([sp] * _n_reg(result_layout()), result_layout()))
        else:
            # Scalar integer (uniform stride / offset) -> symbolic constant.
            bind(Lin({}, operand(0)))
    elif name == "tt.expand_dims":
        bind(_lin_shift(operand(0), op.get_int_attr("axis")))
    elif name == "tt.broadcast":
        # Replication is a no-op on the symbolic form (it adds no coordinate
        # dependence); the consuming op's layout supplies the per-lane spread.
        bind(operand(0))
    elif name == "tt.trans":
        # A physical relabel: the same per-register SIMD slots reinterpreted
        # under the transposed (dim-permuted) layout; no data movement.
        src = operand(0)
        if not isinstance(src, RegVal):
            raise NotImplementedError("wave_amd: tt.trans of a non-data tensor unsupported")
        bind(RegVal(list(src.slots), result_layout()))
    elif name == "ttg.convert_layout":
        src = operand(0)
        if not isinstance(src, RegVal):
            raise NotImplementedError("wave_amd: convert_layout of a non-data tensor unsupported")
        elem = _wave_scalar_type(op.get_result(0).get_type(), dsl)
        simd = dsl.simd_type(elem, width)
        src_ll = op.get_operand(0).get_tensor_layout()
        dst_ll = op.get_result(0).get_tensor_layout()
        bind(_convert_layout(fb, dsl, src, src_ll, dst_ll, elem, simd, width, warp_id))
    elif name == "tt.addptr":
        base, off = operand(0), operand(1)
        bind(PtrVal(base.base, _lin_add(base.lin, off)))
    elif name == "tt.load":
        lay = result_layout()
        rty = dsl.simd_type(_wave_scalar_type(op.get_result(0).get_type(), dsl), width)
        addr = _synth_addr(fb, dsl, operand(0), lay, width, warp_id)
        bind(RegVal([fb.load(addr.slots[r], rty)[0] for r in range(_n_reg(lay))], lay))
    elif name == "tt.store":
        val = operand(1)
        lay = val.layout
        addr = _synth_addr(fb, dsl, operand(0), lay, width, warp_id)
        for r in range(_n_reg(lay)):
            fb.store(_slot(val, r), addr.slots[r])
    else:
        raise NotImplementedError(f"wave_amd: unsupported op {name}")


def convert(ttgir_module, arch: str) -> "ModuleBuilder":  # noqa: F821 (lazy DSL type)
    """Build a Wave-dialect ``ModuleBuilder`` mirroring a finalized TTGIR module.

    ``ttgir_module`` is a finalized TTGIR ``ModuleOp`` (triton ``_C``); ``arch``
    is the gfx target (e.g. ``"gfx1100"``). Returns the ``ModuleBuilder`` so the
    caller can ``bld.module.operation.verify()`` and ``str(bld)``.
    """
    dsl = _import_dsl()
    from mlir.ir import StringAttr

    name = ttgir_module.get_entry_func_name()
    func = ttgir_module.get_function(name)

    ops = []
    ttgir_module.walk(ops.append)

    threads_per_warp = ttgir_module.get_int_attr("ttg.threads-per-warp") or 32
    width = _scope_gate(ops, threads_per_warp, allow_warp=True)

    body = [op for op in ops if op.get_name() not in _SKIP_OPS]
    split = _first_mem_index(body)
    lds_size = _lds_bytes(body)
    multi_warp = _uses_warp(body)
    log2_width = width.bit_length() - 1  # width is a power of two

    bld = dsl.module()
    with bld:
        bld.module.operation.attributes["waveamdmachine.target"] = StringAttr.get(f"amdgcn-amd-amdhsa--{arch}")
        # Wave type builders require the active context entered by `with bld:`.
        inputs = [_wave_scalar_type(func.args(i).get_type(), dsl) for i in range(func.get_num_args())]
        with bld.function(name, inputs, kernel=True, lds_size=lds_size) as fb:
            vmap: dict[int, object] = {}
            for i in range(func.get_num_args()):
                vmap[func.args(i).id()] = fb.args[i]

            # Per-lane warp id (uniform within a wave): workitem_id = warp*W + lane
            # => warp = workitem_id >> log2(W). Masked to log2(num_warps) bits so
            # the address range analyzer can bound it (workitem_id alone infers
            # [0, INT_MAX], which fails the LDS u32 offset check). Materialized
            # only when needed, so single-warp IR (wave.lane_id markers) is
            # unchanged.
            warp_id = None
            if multi_warp:
                num_warps = ttgir_module.get_int_attr("ttg.num-warps") or 1
                bk = dsl.BinaryKind
                warp_id = fb.binary(bk.ShRUI, fb.workitem_id(0), _splat_int(fb, dsl, log2_width, width))
                if num_warps > 1:
                    warp_id = fb.binary(bk.AndI, warp_id, _splat_int(fb, dsl, num_warps - 1, width))

            def run(op):
                _translate_op(op, fb, vmap, width, dsl, warp_id)

            if split is None:
                for op in body:
                    run(op)
            else:
                for op in body[:split]:
                    run(op)
                mask_id = _mask_operand_id(body[split])
                if mask_id is None:
                    for op in body[split:]:
                        run(op)
                else:
                    mask = vmap[mask_id]
                    if isinstance(mask, RegVal) and len(mask.slots) != 1:
                        raise NotImplementedError("wave_amd: masked multi-register memory ops not yet supported")
                    with fb.where(_slot(mask, 0)):
                        for op in body[split:]:
                            run(op)
    return bld


def wave_ir_text(bld) -> str:
    """Return the verified Wave IR assembly text for ``bld``."""
    return str(bld)

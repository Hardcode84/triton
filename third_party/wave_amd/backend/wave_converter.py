"""wave_amd M2: structural finalized-TTGIR -> Wave-dialect converter.

This milestone builds, entirely in one Python process and with no GPU or
assembly, a Wave-dialect MLIR module that structurally mirrors a *finalized*
TTGIR kernel. It reads the TTGIR through triton's ``_C`` bindings (the op walk
plus the typed value/layout/dtype introspection added in M1) and emits Wave IR
through the wave-mlir Python DSL (``mlir.dialects.wave_dsl``), passing only
plain data (op names, dtypes, layout bases, attribute ints) across the two
runtimes' boundary.

Scope -- the single-wave, elementwise masked-copy op class:

  * Single-wave layouts only. Every distributed tensor must keep
    ``register``/``warp``/``block`` on a single element (basis count 0) and
    spread ``lane`` across the wavefront (basis count log2(W), with W equal to
    the module's threads-per-warp). Anything wider raises ``NotImplementedError``.
  * The per-lane wavefront maps to ``!wave.simd<T, W>``. ``tt.make_range`` (the
    ``0..BLOCK`` iota under the lane-identity layout) becomes ``wave.lane_id``.
  * The elementwise mask is hoisted to a region: ops up to the first masked
    memory op are emitted as a flat preamble, and the memory ops are emitted
    inside one ``wave.where(mask)`` region, where the mask is taken from the
    memory op's mask operand.
  * A ``tt.splat`` of a scalar *base pointer* is never materialized; it is fused
    into the following ``tt.addptr`` by aliasing the splat result to the uniform
    Wave pointer, so ``wave.ptr_add(base, offset)`` yields the per-lane
    ``!wave.simd<!wave.ptr<...>, W>`` address directly.

The converter is intentionally small and purely structural: no textual
round-trips and no new C++ -- just a flat, op-by-op translation keyed on SSA
value identity (``Value.id()``).

Public API:
  * ``convert(ttgir_module, arch) -> ModuleBuilder``
  * ``wave_ir_text(bld) -> str``

The Wave DSL is imported lazily inside :func:`convert` (after the wave package
is placed on ``sys.path``) so that the triton-first import order required for
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


def _derive_width(ops, threads_per_warp):
    """Gate single-wave layouts and return the wavefront width W.

    Every tensor result that carries a distributed layout must keep
    register/warp/block on one element and spread lane across the wavefront; all
    such tensors must agree on W, which must equal the module threads-per-warp.
    """
    widths = set()
    for op in ops:
        for i in range(op.get_num_results()):
            ll = op.get_result(i).get_tensor_layout()
            if ll is None:
                continue
            bases = dict(ll.bases)
            for axis in ("register", "warp", "block"):
                if len(bases.get(axis, [])) != 0:
                    raise NotImplementedError(
                        "wave_amd M2: only single-wave (register=warp=block=1) layouts are supported")
            widths.add(2**len(bases.get("lane", [])))
    if not widths:
        return threads_per_warp
    if len(widths) != 1:
        raise NotImplementedError(f"wave_amd M2: tensors disagree on wavefront width: {sorted(widths)}")
    width = widths.pop()
    if width != threads_per_warp:
        raise NotImplementedError(f"wave_amd M2: lane width {width} != threads-per-warp {threads_per_warp}")
    return width


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


def _translate_op(op, fb, vmap, width, dsl):
    """Translate one TTGIR body op into Wave IR, recording results in ``vmap``."""
    name = op.get_name()

    def operand(i):
        return vmap[op.get_operand(i).id()]

    def bind(value):
        vmap[op.get_result(0).id()] = value

    if name == "arith.constant":
        bind(fb.constant(_wave_scalar_type(op.get_result(0).get_type(), dsl), op.get_constant_value()))
    elif name == "tt.get_program_id":
        bind(fb.workgroup_id(0))
    elif name == "arith.muli":
        bind(fb.muli(operand(0), operand(1)))
    elif name == "arith.addi":
        bind(fb.addi(operand(0), operand(1)))
    elif name == "tt.make_range":
        if op.get_int_attr("start") != 0:
            raise NotImplementedError("wave_amd M2: tt.make_range with start != 0 is unsupported")
        bind(fb.lane_id(dsl.i32(), width=width))
    elif name == "tt.splat":
        result = op.get_result(0)
        if result.get_type().is_pointer_type():
            # Scalar base-pointer splat: fuse into the following ptr_add by
            # aliasing the result to the uniform Wave pointer (no op emitted).
            vmap[result.id()] = operand(0)
        else:
            elem = _wave_scalar_type(op.get_operand(0).get_type(), dsl)
            bind(fb.splat(operand(0), elem, width=width))
    elif name == "arith.cmpi":
        bind(fb.cmpi(_CMPI_PRED[op.get_int_attr("predicate")], operand(0), operand(1)))
    elif name == "tt.addptr":
        bind(fb.ptr_add(operand(0), operand(1)))
    elif name == "tt.load":
        rty = dsl.simd_type(_wave_scalar_type(op.get_result(0).get_type(), dsl), width)
        value, _token = fb.load(operand(0), rty)
        bind(value)
    elif name == "tt.store":
        fb.store(operand(1), operand(0))
    else:
        raise NotImplementedError(f"wave_amd M2: unsupported op {name}")


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
    width = _derive_width(ops, threads_per_warp)

    body = [op for op in ops if op.get_name() not in _SKIP_OPS]
    split = _first_mem_index(body)

    bld = dsl.module()
    with bld:
        bld.module.operation.attributes["waveamdmachine.target"] = StringAttr.get(f"amdgcn-amd-amdhsa--{arch}")
        # Wave type builders require the active context entered by `with bld:`.
        inputs = [_wave_scalar_type(func.args(i).get_type(), dsl) for i in range(func.get_num_args())]
        with bld.function(name, inputs, kernel=True) as fb:
            vmap: dict[int, object] = {}
            for i in range(func.get_num_args()):
                vmap[func.args(i).id()] = fb.args[i]

            if split is None:
                for op in body:
                    _translate_op(op, fb, vmap, width, dsl)
            else:
                for op in body[:split]:
                    _translate_op(op, fb, vmap, width, dsl)
                mask_id = _mask_operand_id(body[split])
                if mask_id is None:
                    for op in body[split:]:
                        _translate_op(op, fb, vmap, width, dsl)
                else:
                    with fb.where(vmap[mask_id]):
                        for op in body[split:]:
                            _translate_op(op, fb, vmap, width, dsl)
    return bld


def wave_ir_text(bld) -> str:
    """Return the verified Wave IR assembly text for ``bld``."""
    return str(bld)

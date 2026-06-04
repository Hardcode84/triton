import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


@dataclass
class _Value:
    wave: object = None
    elem_type: Optional[str] = None
    const: Optional[int] = None
    expr: object = None
    bindings: Dict[object, object] = field(default_factory=dict)
    index: object = None
    token: object = None
    ptr_base: object = None
    mask_id: Optional[int] = None
    mask_wave: object = None
    other_wave: object = None
    other_splat_const: Optional[Tuple[object, str, Optional[int]]] = None
    splat_const: Optional[Tuple[object, str, Optional[int]]] = None


def lower_ttir_to_wave_mlir(src, options) -> Tuple[str, str]:
    """Lower a small TTIR slice from the live MLIR module to Wave MLIR.

    This does not parse TTIR assembly. It walks the libtriton MLIR module,
    translates supported ops by operands/results, and emits Wave ops through
    the Wave Python builder DSL.
    """

    if isinstance(src, str):
        raise TypeError("wave_amd lowering expects a TTIR MLIR module object, not textual TTIR")
    lowerer = _TTIRToWaveLowerer(src, options)
    return lowerer.lower()


class _TTIRToWaveLowerer:

    def __init__(self, module, options) -> None:
        self.module = module
        self.options = options
        self.dsl = _load_wave_dsl()
        self.width = int(options.warp_size)
        self.values: Dict[int, _Value] = {}
        self.func = None
        self.load_tokens = []
        self.symbols: Dict[str, object] = {}

    def lower(self) -> Tuple[str, str]:
        name = self.module.get_entry_func_name()
        if not name:
            raise NotImplementedError("wave_amd expects a kernel tt.func entry point")

        ops = self._collect_entry_ops(name)
        func_op = self.module.get_function(name)
        target = f"amdgcn-amd-amdhsa--{self.options.arch}"

        with self.dsl.module() as module_builder:
            arg_types = [self._signature_type(ty) for ty in self.module.get_function_signature(func_op)]
            del module_builder.module.operation.attributes["gpu.container_module"]
            module_builder.module.operation.attributes["waveamdmachine.target"] = self.dsl.StringAttr.get(target)
            with module_builder.function(name, arg_types, kernel=True) as func:
                self.func = func
                self._bind_arguments(func_op, func.args)
                self._lower_ops(ops)
            return str(module_builder), name

    def _collect_entry_ops(self, entry_name: str) -> Sequence[object]:
        pending_body_ops = []
        entry_ops = []

        def visit(op):
            name = op.get_name()
            if name == "builtin.module":
                return
            if name == "tt.func":
                nonlocal entry_ops
                if op.get_str_attr("sym_name") == entry_name:
                    entry_ops = list(pending_body_ops)
                pending_body_ops.clear()
                return
            pending_body_ops.append(op)

        self.module.walk(visit)
        return entry_ops

    def _bind_arguments(self, func_op, wave_args) -> None:
        signatures = self.module.get_function_signature(func_op)
        for index in range(func_op.get_num_args()):
            signature = signatures[index]
            state = _Value(wave=wave_args[index], elem_type=_signature_element_type(signature))
            if signature.startswith("*"):
                state.ptr_base = wave_args[index]
            self.values[func_op.args(index).id()] = state

    def _lower_ops(self, ops: Sequence[object]) -> None:
        index = 0
        while index < len(ops):
            op = ops[index]
            name = op.get_name()
            if name == "tt.return":
                return
            if name in {"builtin.module", "tt.func", "llvm.intr.assume"}:
                index += 1
                continue

            self._lower_op(op)
            index += 1

    def _lower_op(self, op) -> None:
        name = op.get_name()
        if name == "arith.constant":
            self._lower_constant(op)
        elif name == "arith.addi":
            self._lower_binary(op, lambda lhs, rhs: lhs + rhs, self.func.addi)
        elif name == "arith.muli":
            self._lower_binary(op, lambda lhs, rhs: lhs * rhs, self.func.muli)
        elif name == "arith.addf":
            self._lower_binary(op, None, self.func.fadd)
        elif name == "arith.cmpi":
            self._lower_cmpi(op)
        elif name == "tt.get_program_id":
            self._lower_program_id(op)
        elif name == "tt.make_range":
            self._lower_make_range(op)
        elif name == "tt.splat":
            self._lower_splat(op)
        elif name == "tt.addptr":
            self._lower_addptr(op)
        elif name == "tt.load":
            self._lower_load(op)
        elif name == "tt.store":
            self._lower_store(op)
        else:
            raise NotImplementedError(f"wave_amd TTIR lowering does not support op: {name}")

    def _lower_constant(self, op) -> None:
        constant = self._arith_constant_splat(op)
        state = _Value()
        if constant is not None:
            value, elem_type, width = constant
            scalar = self.func.constant(self._scalar_type(elem_type), value)
            state.elem_type = elem_type
            state.const = value if isinstance(value, int) and width is None else None
            state.splat_const = (value, elem_type, width)
            if width is None:
                state.wave = scalar
            else:
                if width != self.width:
                    raise NotImplementedError(
                        f"wave_amd only supports splat constants with width {self.width}, got {width}")
                state.wave = self.func.splat(scalar, self._scalar_type(elem_type), width)
        self._set_result(op, state)

    def _lower_program_id(self, op) -> None:
        axis = self._program_id_axis(op)
        if axis is None:
            raise NotImplementedError(
                "wave_amd cannot lower tt.get_program_id until ProgramDimAttr is exposed structurally")
        pid = self.func.workgroup_id(axis)
        sym = self._sym(f"pid_{axis}")
        self._set_result(op, _Value(wave=pid, elem_type="i32", expr=sym, bindings={sym: pid}))

    def _lower_make_range(self, op) -> None:
        start = op.get_int_attr("start")
        end = op.get_int_attr("end")
        if start != 0 or end != self.width:
            raise NotImplementedError(
                f"wave_amd only supports tt.make_range {{start = 0, end = warp_size}}, got {start}:{end}")

        lane = self.func.lane_id(self.dsl.i32(), self.width)
        sym = self._sym("lid")
        self._set_result(
            op,
            _Value(
                wave=lane,
                elem_type="i32",
                expr=sym,
                bindings={sym: lane},
            ),
        )

    def _lower_splat(self, op) -> None:
        src = self._value(op.get_operand(0))
        if src.ptr_base is not None:
            state = _Value(wave=src.ptr_base, elem_type=src.elem_type, ptr_base=src.ptr_base, mask_id=src.mask_id)
        else:
            splat_const = None
            if src.splat_const is not None and src.splat_const[2] is None:
                splat_const = (src.splat_const[0], src.splat_const[1], self.width)
            state = _Value(
                wave=self.func.splat(src.wave, self._scalar_type(src.elem_type), self.width),
                elem_type=src.elem_type,
                expr=src.expr,
                bindings=dict(src.bindings),
                mask_id=src.mask_id,
                splat_const=splat_const,
            )
        self._set_result(op, state)

    def _lower_binary(self, op, expr_builder, wave_builder) -> None:
        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        mask_id = _merge_mask_ids(lhs, rhs)
        wave = wave_builder(lhs.wave, rhs.wave)
        expr = None
        bindings = {}
        if expr_builder is not None:
            lhs_expr, lhs_bindings = self._expr_and_bindings(lhs)
            rhs_expr, rhs_bindings = self._expr_and_bindings(rhs)
            if lhs_expr is not None and rhs_expr is not None:
                expr = expr_builder(lhs_expr, rhs_expr)
                bindings = {**lhs_bindings, **rhs_bindings}
        self._set_result(
            op,
            _Value(
                wave=wave,
                elem_type=lhs.elem_type or rhs.elem_type,
                expr=expr,
                bindings=bindings,
                mask_id=mask_id,
            ),
        )

    def _lower_cmpi(self, op) -> None:
        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        predicate = self._cmpi_predicate(op)
        if predicate is None:
            raise NotImplementedError(
                "wave_amd cannot lower arith.cmpi until the predicate attribute is exposed structurally")
        self._set_result(op, _Value(wave=self.func.cmpi(predicate, lhs.wave, rhs.wave), elem_type="i1"))

    def _lower_addptr(self, op) -> None:
        ptr = self._value(op.get_operand(0))
        offset = self._value(op.get_operand(1))
        index = self._index(offset)
        self._set_result(
            op,
            _Value(
                wave=self.func.ptr_add(ptr.ptr_base or ptr.wave, index),
                elem_type=ptr.elem_type,
            ),
        )

    def _lower_load(self, op) -> None:
        mask = self._memory_mask(op)
        if mask is not None:
            if op.get_num_operands() not in {2, 3}:
                raise NotImplementedError("wave_amd masked tt.load supports pointer, mask, and optional other only")
            ptr = self._value(op.get_operand(0))
            mask_value = self._value(mask)
            if mask_value.wave is None:
                raise NotImplementedError("wave_amd masked load requires a Wave mask value")
            result_type = self._load_result_type(ptr)
            other = None
            if op.get_num_operands() == 3:
                other = self._load_other_value(self._value(op.get_operand(2)), ptr.elem_type)
            with self.func.where(mask_value.wave, [result_type, self.dsl.mem_token_type()]) as where_op:
                value, token = self._emit_load(ptr, result_type)
                self.func.yield_([value, token])
            value, token = where_op.results
            if other is not None:
                self._set_result(
                    op,
                    _Value(
                        wave=value,
                        elem_type=ptr.elem_type,
                        token=token,
                        mask_id=mask.id(),
                        mask_wave=mask_value.wave,
                        other_wave=other.wave,
                        other_splat_const=other.splat_const,
                    ),
                )
            else:
                self.load_tokens.append(token)
                self._set_result(op, _Value(wave=value, elem_type=ptr.elem_type, token=token, mask_id=mask.id()))
            return

        ptr = self._value(op.get_operand(0))
        value, token = self._emit_load(ptr)
        self.load_tokens.append(token)
        self._set_result(op, _Value(wave=value, elem_type=ptr.elem_type, token=token))

    def _lower_store(self, op) -> None:
        ptr = self._value(op.get_operand(0))
        value = self._value(op.get_operand(1))
        mask = self._memory_mask(op)
        if mask is None and value.other_wave is not None and value.mask_wave is not None:
            self._emit_unmasked_store_of_masked_load_other(ptr, value)
            return
        if mask is not None:
            if op.get_num_operands() != 3:
                raise NotImplementedError("wave_amd masked tt.store supports pointer, value, and mask operands only")
            if value.mask_id is not None and value.mask_id != mask.id():
                raise NotImplementedError(
                    "wave_amd masked load values must be stored under the same SSA mask that produced them")
            mask_value = self._value(mask)
            if mask_value.wave is None:
                raise NotImplementedError("wave_amd masked store requires a Wave mask value")
            with self.func.where(mask_value.wave):
                self._emit_store(ptr, value)
            self.load_tokens.clear()
            return

        if value.mask_id is not None:
            raise NotImplementedError("wave_amd masked load value cannot be stored without its producing SSA mask")
        self._emit_store(ptr, value)

    def _load_result_type(self, ptr: _Value):
        elem_type = ptr.elem_type
        return self.dsl.simd_type(self._scalar_type(elem_type), self.width)

    def _emit_load(self, ptr: _Value, result_type=None):
        return self.func.load(ptr.wave, result_type or self._load_result_type(ptr))

    def _emit_store(self, ptr: _Value, value: _Value) -> None:
        after = None
        if len(self.load_tokens) == 1:
            after = self.load_tokens[0]
        elif len(self.load_tokens) > 1:
            after = self.func.join(*self.load_tokens)
        self.func.store(value.wave, ptr.wave, after=after)
        self.load_tokens.clear()

    def _emit_unmasked_store_of_masked_load_other(self, ptr: _Value, value: _Value) -> None:
        with self.func.where(value.mask_wave, [self.dsl.mem_token_type()]) as where_op:
            token = self.func.store(value.wave, ptr.wave, after=value.token)
            self.func.yield_([token])
        block = where_op.elseRegion.blocks.append()
        with self.dsl.InsertionPoint(block):
            other = value.other_wave
            if value.other_splat_const is not None:
                other = self._materialize_splat_constant(value.other_splat_const)
            token = self.func.store(other, ptr.wave)
            self.dsl.wave.YieldOp([token])
        self.load_tokens.clear()

    def _load_other_value(self, other: _Value, elem_type: str) -> _Value:
        if other.wave is None:
            raise NotImplementedError("wave_amd masked tt.load `other` requires a Wave value")
        if str(other.wave.type).startswith("!wave.simd<"):
            return other
        splat = self.func.splat(other.wave, self._scalar_type(other.elem_type or elem_type), self.width)
        splat_const = None
        if other.splat_const is not None and other.splat_const[2] is None:
            splat_const = (other.splat_const[0], other.splat_const[1], self.width)
        return _Value(wave=splat, elem_type=other.elem_type or elem_type, splat_const=splat_const)

    def _materialize_splat_constant(self, constant: Tuple[object, str, Optional[int]]):
        value, elem_type, width = constant
        scalar = self.func.constant(self._scalar_type(elem_type), value)
        if width is None:
            return scalar
        return self.func.splat(scalar, self._scalar_type(elem_type), width)

    def _memory_mask(self, op):
        name = op.get_name()
        if name == "tt.load" and op.get_num_operands() >= 2:
            return op.get_operand(1)
        if name == "tt.store" and op.get_num_operands() >= 3:
            return op.get_operand(2)
        return None

    def _index(self, state: _Value):
        if state.index is not None:
            return state.index
        expr, bindings = self._expr_and_bindings(state)
        if expr is None:
            raise NotImplementedError("wave_amd requires symbolic tt.addptr offsets")
        return self.func.index_expr(expr, bindings, self.dsl.simd_type(self.dsl.index_type(), self.width))

    def _expr_and_bindings(self, state: _Value):
        if state.expr is not None:
            return state.expr, dict(state.bindings)
        if state.const is None:
            return None, {}

        name = f"c{state.const}"
        sym = self._sym(name)
        return sym, {sym: state.wave}

    def _program_id_axis(self, op) -> Optional[int]:
        axis = _wave_amd_native().get_program_id_axis(op)
        return None if axis is None else int(axis)

    def _cmpi_predicate(self, op) -> Optional[str]:
        predicate = _wave_amd_native().get_cmpi_predicate(op)
        return None if predicate is None else str(predicate)

    def _arith_constant_splat(self, op):
        constant = _wave_amd_native().get_arith_constant_splat(op)
        if constant is None:
            return None
        value, elem_type, width = constant
        return value, str(elem_type), None if width is None else int(width)

    def _sym(self, name: str):
        if name not in self.symbols:
            self.symbols[name] = self.dsl.sym(name)
        return self.symbols[name]

    def _value(self, value) -> _Value:
        try:
            return self.values[value.id()]
        except KeyError as exc:
            raise NotImplementedError("wave_amd encountered a value produced by an unsupported TTIR op") from exc

    def _set_result(self, op, state: _Value) -> None:
        if op.get_num_results() != 1:
            raise NotImplementedError(f"wave_amd expected one result from {op.get_name()}")
        self.values[op.get_result(0).id()] = state

    def _signature_type(self, signature: str):
        if signature.startswith("*"):
            return self.dsl.ptr_type(self._scalar_type(signature[1:]), self.dsl.global_address_space())
        return self._scalar_type(signature)

    def _scalar_type(self, name: Optional[str]):
        if name == "i1":
            return self.dsl.i1()
        if name == "i8":
            return self.dsl.i8()
        if name == "i32":
            return self.dsl.i32()
        if name == "i64":
            return self.dsl.i64()
        if name == "f16":
            return self.dsl.f16()
        if name == "bf16":
            return self.dsl.bf16()
        if name == "f32":
            return self.dsl.f32()
        if name == "index":
            return self.dsl.index_type()
        raise NotImplementedError(f"wave_amd does not support type {name!r}")


def _signature_element_type(signature: str) -> Optional[str]:
    if signature.startswith("*"):
        return signature[1:]
    return signature


def _merge_mask_ids(*states: _Value) -> Optional[int]:
    mask_ids = {state.mask_id for state in states if state.mask_id is not None}
    if not mask_ids:
        return None
    if len(mask_ids) != 1:
        raise NotImplementedError("wave_amd masked values must share one SSA mask")
    return next(iter(mask_ids))


def _load_wave_dsl():
    try:
        from mlir.dialects import wave_dsl
        return wave_dsl
    except ImportError as first_error:
        wave_python = Path(__file__).resolve().parents[1] / "wave" / "python"
        if wave_python.is_dir() and str(wave_python) not in sys.path:
            sys.path.insert(0, str(wave_python))
        try:
            from mlir.dialects import wave_dsl
            return wave_dsl
        except ImportError as exc:
            cause = first_error if exc is not first_error else exc
            raise RuntimeError(
                "wave_amd lowering requires the Wave Python MLIR builder bindings. "
                "Build/install the Wave submodule Python bindings so `mlir.dialects.wave_dsl` "
                "can be imported; this backend intentionally does not fall back to textual MLIR assembly.") from cause


def _wave_amd_native():
    try:
        from triton._C.libtriton import wave_amd
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "wave_amd lowering requires the Triton wave_amd native extension. "
            "Rebuild Triton with the wave_amd backend so TTIR operation attributes can be read structurally.") from exc

    required = ("get_program_id_axis", "get_cmpi_predicate", "get_arith_constant_splat")
    missing = [name for name in required if not hasattr(wave_amd, name)]
    if missing:
        raise RuntimeError("wave_amd lowering requires the Triton wave_amd native extension to expose "
                           f"{', '.join(missing)}. Rebuild Triton with the updated wave_amd backend.")
    return wave_amd

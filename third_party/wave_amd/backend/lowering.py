import sys
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


def _product(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class _TensorInfo:
    shape: Tuple[int, ...]
    elem_type: str
    is_pointer: bool = False


@dataclass(frozen=True)
class _BlockedLayout:
    shape: Tuple[int, ...]
    width: int
    num_warps: int
    registers: int

    @classmethod
    def for_tensor(cls, info: _TensorInfo, width: int, num_warps: int, num_ctas: int) -> "_BlockedLayout":
        if num_warps < 1 or not _is_power_of_two(num_warps):
            raise NotImplementedError("wave_amd layout lowering requires power-of-two num_warps")
        if num_ctas != 1:
            raise NotImplementedError("wave_amd general layout lowering currently supports one CTA per CGA")
        elements = _product(info.shape)
        if any(not _is_power_of_two(dim) for dim in info.shape):
            raise NotImplementedError("wave_amd layout lowering requires power-of-two tensor dimensions")
        if not _is_power_of_two(elements):
            raise NotImplementedError("wave_amd layout lowering requires power-of-two tensor shapes")
        threads = width * num_warps
        if elements < threads or elements % threads != 0:
            raise NotImplementedError("wave_amd layout lowering requires tensor elements to be a multiple of "
                                      f"num_warps * warp_size, got {elements}")
        return cls(shape=info.shape, width=width, num_warps=num_warps, registers=elements // threads)

    def index_expr(self, dsl, lane_sym, register: int):
        if register < 0 or register >= self.registers:
            raise IndexError(register)
        if register == 0:
            return lane_sym
        return lane_sym + register * self.width * self.num_warps

    def coord_expr(self, dsl, lane_sym, register: int, axis: int):
        if axis < 0 or axis >= len(self.shape):
            raise IndexError(axis)
        flat = self.index_expr(dsl, lane_sym, register)
        stride = _product(self.shape[axis + 1:])
        if stride != 1:
            flat = dsl.floor(flat / stride)
        dim = self.shape[axis]
        if dim == 1:
            return dsl.sym_ctx.int_(0)
        return dsl.mod(flat, dim)


@dataclass
class _Value:
    wave: object = None
    waves: Tuple[object, ...] = field(default_factory=tuple)
    elem_type: Optional[str] = None
    const: Optional[int] = None
    expr: object = None
    exprs: Tuple[object, ...] = field(default_factory=tuple)
    bindings: Dict[object, object] = field(default_factory=dict)
    bindings_by_wave: Tuple[Dict[object, object], ...] = field(default_factory=tuple)
    index: object = None
    indexes: Tuple[object, ...] = field(default_factory=tuple)
    token: object = None
    tokens: Tuple[object, ...] = field(default_factory=tuple)
    ptr_base: object = None
    mask_id: Optional[int] = None
    mask_wave: object = None
    mask_waves: Tuple[object, ...] = field(default_factory=tuple)
    other_wave: object = None
    other_waves: Tuple[object, ...] = field(default_factory=tuple)
    other_splat_const: Optional[Tuple[object, str, Optional[int]]] = None
    splat_const: Optional[Tuple[object, str, Optional[int]]] = None
    layout: Optional[_BlockedLayout] = None
    shape: Optional[Tuple[int, ...]] = None
    coord_axis: Optional[int] = None
    coord_start: int = 0


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
        self.num_warps = int(getattr(options, "num_warps", 1))
        self.num_ctas = int(getattr(options, "num_ctas", 1))
        self.values: Dict[int, _Value] = {}
        self.func = None
        self.load_tokens = []
        self.symbols: Dict[str, object] = {}
        self._lane_value = None
        self._workitem_value = None

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
        elif name == "arith.subi":
            self._lower_subi(op)
        elif name == "arith.muli":
            self._lower_binary(op, lambda lhs, rhs: lhs * rhs, self.func.muli)
        elif name == "arith.addf":
            self._lower_binary(op, None, self.func.fadd)
        elif name == "arith.subf":
            self._lower_binary(op, None, self.func.fsub)
        elif name == "arith.mulf":
            self._lower_binary(op, None, self.func.fmul)
        elif name == "arith.cmpi":
            self._lower_cmpi(op)
        elif name == "arith.select":
            self._lower_select(op)
        elif name == "tt.get_program_id":
            self._lower_program_id(op)
        elif name == "tt.make_range":
            self._lower_make_range(op)
        elif name == "tt.expand_dims":
            self._lower_expand_dims(op)
        elif name == "tt.broadcast":
            self._lower_broadcast(op)
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
                info = self._result_tensor_info(op)
                if info is None:
                    raise NotImplementedError("wave_amd dense constants require tensor result type metadata")
                layout = self._layout_for_info(info)
                waves = tuple(
                    self.func.splat(scalar, self._scalar_type(elem_type), self.width) for _ in range(layout.registers))
                state.wave = waves[0]
                state.waves = waves
                state.layout = layout
                state.shape = info.shape
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
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.make_range requires tensor result type metadata")
        if len(info.shape) != 1 or info.shape[0] != end - start:
            raise NotImplementedError("wave_amd tt.make_range shape must match end - start")

        layout = self._layout_for_info(info)
        state = self._coordinate_value(layout, info, axis=0, start=start)
        self._set_result(
            op,
            _Value(
                wave=state.wave,
                waves=state.waves,
                elem_type="i32",
                expr=state.expr,
                exprs=state.exprs,
                bindings=state.bindings,
                bindings_by_wave=state.bindings_by_wave,
                layout=layout,
                shape=info.shape,
                coord_axis=0,
                coord_start=start,
            ),
        )

    def _lower_expand_dims(self, op) -> None:
        src = self._value(op.get_operand(0))
        axis = op.get_int_attr("axis")
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.expand_dims requires tensor result type metadata")
        if src.coord_axis is None:
            raise NotImplementedError("wave_amd tt.expand_dims currently supports coordinate tensors")
        coord_axis = src.coord_axis + 1 if axis <= src.coord_axis else src.coord_axis
        layout = self._layout_for_info(info)
        state = self._coordinate_value(layout, _TensorInfo(info.shape, src.elem_type or info.elem_type), coord_axis,
                                       src.coord_start)
        self._set_result(op, state)

    def _lower_broadcast(self, op) -> None:
        src = self._value(op.get_operand(0))
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.broadcast requires tensor result type metadata")
        layout = self._layout_for_info(info)
        if src.coord_axis is not None:
            state = self._coordinate_value(layout, _TensorInfo(info.shape, src.elem_type or info.elem_type),
                                           src.coord_axis, src.coord_start)
        else:
            state = self._coerce_tensor(src, layout, info)
        self._set_result(op, state)

    def _lower_splat(self, op) -> None:
        src = self._value(op.get_operand(0))
        info = self._result_tensor_info(op)
        layout = self._layout_for_info(info) if info is not None else None
        if src.ptr_base is not None:
            state = _Value(
                wave=src.ptr_base,
                elem_type=src.elem_type,
                ptr_base=src.ptr_base,
                mask_id=src.mask_id,
                layout=layout,
                shape=None if info is None else info.shape,
            )
        else:
            expr, bindings = self._expr_and_bindings(src)
            splat_const = None
            if src.splat_const is not None and src.splat_const[2] is None and layout is not None:
                splat_const = (src.splat_const[0], src.splat_const[1], _product(info.shape))
            if layout is not None:
                waves = tuple(
                    self.func.splat(src.wave, self._scalar_type(src.elem_type), self.width)
                    for _ in range(layout.registers))
                exprs = tuple(expr for _ in range(layout.registers))
                bindings_by_wave = tuple(dict(bindings) for _ in range(layout.registers))
                state = _Value(
                    wave=waves[0],
                    waves=waves,
                    elem_type=src.elem_type,
                    const=src.const,
                    expr=expr,
                    exprs=exprs,
                    bindings=bindings,
                    bindings_by_wave=bindings_by_wave,
                    mask_id=src.mask_id,
                    splat_const=splat_const,
                    layout=layout,
                    shape=info.shape,
                )
            else:
                wave = self.func.splat(src.wave, self._scalar_type(src.elem_type), self.width)
                state = _Value(
                    wave=wave,
                    waves=(wave, ),
                    elem_type=src.elem_type,
                    const=src.const,
                    expr=expr,
                    exprs=(expr, ),
                    bindings=bindings,
                    bindings_by_wave=(bindings, ),
                    mask_id=src.mask_id,
                    splat_const=splat_const,
                )
        self._set_result(op, state)

    def _lower_binary(self, op, expr_builder, wave_builder) -> None:
        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        mask_id = _merge_mask_ids(lhs, rhs)
        info = self._result_tensor_info(op)
        if info is not None:
            layout = self._layout_for_info(info)
            lhs = self._coerce_tensor(lhs, layout, info)
            rhs = self._coerce_tensor(rhs, layout, info)
            waves = ()
            if lhs.waves and rhs.waves:
                waves = tuple(wave_builder(lhs_wave, rhs_wave) for lhs_wave, rhs_wave in zip(lhs.waves, rhs.waves))
            exprs = []
            bindings_by_wave = []
            if expr_builder is not None:
                for index in range(layout.registers):
                    lhs_expr, lhs_bindings = self._expr_and_bindings(lhs, index)
                    rhs_expr, rhs_bindings = self._expr_and_bindings(rhs, index)
                    if lhs_expr is not None and rhs_expr is not None:
                        exprs.append(expr_builder(lhs_expr, rhs_expr))
                        bindings_by_wave.append({**lhs_bindings, **rhs_bindings})
                    else:
                        exprs.append(None)
                        bindings_by_wave.append({})
            else:
                exprs = [None] * layout.registers
                bindings_by_wave = [{} for _ in range(layout.registers)]
            self._set_result(
                op,
                _Value(
                    wave=waves[0] if waves else None,
                    waves=waves,
                    elem_type=lhs.elem_type or rhs.elem_type,
                    expr=exprs[0],
                    exprs=tuple(exprs),
                    bindings=bindings_by_wave[0],
                    bindings_by_wave=tuple(bindings_by_wave),
                    mask_id=mask_id,
                    layout=layout,
                    shape=info.shape,
                ),
            )
            return

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

    def _lower_subi(self, op) -> None:
        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        mask_id = _merge_mask_ids(lhs, rhs)
        info = self._result_tensor_info(op)
        if info is not None:
            layout = self._layout_for_info(info)
            lhs = self._coerce_tensor(lhs, layout, info)
            rhs = self._coerce_tensor(rhs, layout, info)
            waves = []
            exprs = []
            bindings_by_wave = []
            for index in range(layout.registers):
                if lhs.waves and rhs.waves:
                    lhs_wave = lhs.waves[index]
                    rhs_wave = rhs.waves[index]
                    neg_rhs = self.func.muli(rhs_wave, self._negative_one_like(rhs, index))
                    waves.append(self.func.addi(lhs_wave, neg_rhs))
                lhs_expr, lhs_bindings = self._expr_and_bindings(lhs, index)
                rhs_expr, rhs_bindings = self._expr_and_bindings(rhs, index)
                if lhs_expr is not None and rhs_expr is not None:
                    exprs.append(lhs_expr - rhs_expr)
                    bindings_by_wave.append({**lhs_bindings, **rhs_bindings})
                else:
                    exprs.append(None)
                    bindings_by_wave.append({})
            self._set_result(
                op,
                _Value(
                    wave=waves[0] if waves else None,
                    waves=tuple(waves),
                    elem_type=lhs.elem_type or rhs.elem_type,
                    expr=exprs[0],
                    exprs=tuple(exprs),
                    bindings=bindings_by_wave[0],
                    bindings_by_wave=tuple(bindings_by_wave),
                    mask_id=mask_id,
                    layout=layout,
                    shape=info.shape,
                ),
            )
            return

        neg_rhs = self.func.muli(rhs.wave, self._negative_one_like(rhs))
        wave = self.func.addi(lhs.wave, neg_rhs)
        expr = None
        bindings = {}
        lhs_expr, lhs_bindings = self._expr_and_bindings(lhs)
        rhs_expr, rhs_bindings = self._expr_and_bindings(rhs)
        if lhs_expr is not None and rhs_expr is not None:
            expr = lhs_expr - rhs_expr
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
        info = self._result_tensor_info(op)
        if info is not None:
            layout = self._layout_for_info(info)
            lhs = self._coerce_tensor(lhs, layout, info)
            rhs = self._coerce_tensor(rhs, layout, info)
            waves = tuple(
                self.func.cmpi(predicate, lhs_wave, rhs_wave) for lhs_wave, rhs_wave in zip(lhs.waves, rhs.waves))
            self._set_result(
                op,
                _Value(
                    wave=waves[0],
                    waves=waves,
                    elem_type="i1",
                    layout=layout,
                    shape=info.shape,
                ),
            )
            return
        self._set_result(op, _Value(wave=self.func.cmpi(predicate, lhs.wave, rhs.wave), elem_type="i1"))

    def _lower_select(self, op) -> None:
        condition = self._value(op.get_operand(0))
        true_value = self._value(op.get_operand(1))
        false_value = self._value(op.get_operand(2))
        info = self._result_tensor_info(op)
        if info is not None:
            layout = self._layout_for_info(info)
            condition = self._coerce_tensor(condition, layout, _TensorInfo(info.shape, "i1"))
            true_value = self._coerce_tensor(true_value, layout, info)
            false_value = self._coerce_tensor(false_value, layout, info)
            waves = tuple(
                self.func.select(cond_wave, true_wave, false_wave)
                for cond_wave, true_wave, false_wave in zip(condition.waves, true_value.waves, false_value.waves))
            self._set_result(
                op,
                _Value(
                    wave=waves[0],
                    waves=waves,
                    elem_type=true_value.elem_type or false_value.elem_type,
                    mask_id=_merge_mask_ids(true_value, false_value),
                    layout=layout,
                    shape=info.shape,
                ),
            )
            return
        true_wave, false_wave = self._select_operands(true_value, false_value)
        self._set_result(
            op,
            _Value(
                wave=self.func.select(condition.wave, true_wave, false_wave),
                elem_type=true_value.elem_type or false_value.elem_type,
                mask_id=_merge_mask_ids(true_value, false_value),
            ),
        )

    def _lower_addptr(self, op) -> None:
        ptr = self._value(op.get_operand(0))
        offset = self._value(op.get_operand(1))
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.addptr requires tensor result type metadata")
        layout = self._layout_for_info(info)
        offset = self._coerce_tensor(offset, layout, _TensorInfo(info.shape, "i32"))
        indexes = self._indexes(offset, layout)
        base_waves = ptr.waves if ptr.waves else tuple(ptr.wave for _ in range(layout.registers))
        if ptr.ptr_base is not None:
            base_waves = tuple(ptr.ptr_base for _ in range(layout.registers))
        if len(base_waves) != layout.registers:
            raise NotImplementedError("wave_amd tt.addptr pointer and offset layouts must match")
        waves = tuple(self.func.ptr_add(base, index) for base, index in zip(base_waves, indexes))
        self._set_result(
            op,
            _Value(
                wave=waves[0],
                waves=waves,
                elem_type=ptr.elem_type,
                layout=layout,
                shape=info.shape,
            ),
        )

    def _lower_load(self, op) -> None:
        mask = self._memory_mask(op)
        if mask is not None:
            if op.get_num_operands() not in {2, 3}:
                raise NotImplementedError("wave_amd masked tt.load supports pointer, mask, and optional other only")
            ptr = self._value(op.get_operand(0))
            mask_value = self._value(mask)
            if mask_value.wave is None and not mask_value.waves:
                raise NotImplementedError("wave_amd masked load requires a Wave mask value")
            layout = ptr.layout or self._layout_for_info(self._result_tensor_info(op))
            mask_value = self._coerce_tensor(mask_value, layout, _TensorInfo(ptr.shape or layout.shape, "i1"))
            other = None
            if op.get_num_operands() == 3:
                other = self._load_other_value(self._value(op.get_operand(2)), ptr.elem_type)
                other = self._coerce_tensor(other, layout, _TensorInfo(ptr.shape or layout.shape, ptr.elem_type))
            values = []
            tokens = []
            for index, (ptr_wave, mask_wave) in enumerate(zip(self._waves(ptr), mask_value.waves)):
                result_type = self._load_result_type(ptr)
                with self.func.where(mask_wave, [result_type, self.dsl.mem_token_type()]) as where_op:
                    value, token = self._emit_load_wave(ptr_wave, result_type)
                    self.func.yield_([value, token])
                value, token = where_op.results
                values.append(value)
                tokens.append(token)
            if other is not None:
                self._set_result(
                    op,
                    _Value(
                        wave=values[0],
                        waves=tuple(values),
                        elem_type=ptr.elem_type,
                        token=tokens[0],
                        tokens=tuple(tokens),
                        mask_id=mask.id(),
                        mask_wave=mask_value.waves[0],
                        mask_waves=mask_value.waves,
                        other_wave=other.waves[0],
                        other_waves=other.waves,
                        other_splat_const=other.splat_const,
                        layout=layout,
                        shape=ptr.shape,
                    ),
                )
            else:
                self.load_tokens.extend(tokens)
                self._set_result(
                    op,
                    _Value(
                        wave=values[0],
                        waves=tuple(values),
                        elem_type=ptr.elem_type,
                        token=tokens[0],
                        tokens=tuple(tokens),
                        mask_id=mask.id(),
                        layout=layout,
                        shape=ptr.shape,
                    ),
                )
            return

        ptr = self._value(op.get_operand(0))
        values = []
        tokens = []
        for ptr_wave in self._waves(ptr):
            value, token = self._emit_load_wave(ptr_wave, self._load_result_type(ptr))
            values.append(value)
            tokens.append(token)
        self.load_tokens.extend(tokens)
        self._set_result(
            op,
            _Value(
                wave=values[0],
                waves=tuple(values),
                elem_type=ptr.elem_type,
                token=tokens[0],
                tokens=tuple(tokens),
                layout=ptr.layout,
                shape=ptr.shape,
            ),
        )

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
            if mask_value.wave is None and not mask_value.waves:
                raise NotImplementedError("wave_amd masked store requires a Wave mask value")
            layout = ptr.layout or value.layout
            mask_value = self._coerce_tensor(mask_value, layout, _TensorInfo(ptr.shape or value.shape, "i1"))
            value = self._coerce_tensor(value, layout, _TensorInfo(ptr.shape or value.shape, value.elem_type))
            for mask_wave, ptr_wave, value_wave in zip(mask_value.waves, self._waves(ptr), value.waves):
                with self.func.where(mask_wave):
                    self._emit_store_wave(ptr_wave, value_wave)
            self.load_tokens.clear()
            return

        if value.mask_id is not None:
            raise NotImplementedError("wave_amd masked load value cannot be stored without its producing SSA mask")
        self._emit_store(ptr, value)

    def _load_result_type(self, ptr: _Value):
        elem_type = ptr.elem_type
        return self.dsl.simd_type(self._scalar_type(elem_type), self.width)

    def _emit_load_wave(self, ptr_wave, result_type):
        return self.func.load(ptr_wave, result_type)

    def _emit_store(self, ptr: _Value, value: _Value) -> None:
        if ptr.layout is not None:
            value_info = _TensorInfo(ptr.shape, value.elem_type)
            value = self._coerce_tensor(value, ptr.layout, value_info)
        for ptr_wave, value_wave in zip(self._waves(ptr), self._waves(value)):
            self._emit_store_wave(ptr_wave, value_wave)
        self.load_tokens.clear()

    def _emit_store_wave(self, ptr_wave, value_wave) -> None:
        after = None
        if len(self.load_tokens) == 1:
            after = self.load_tokens[0]
        elif len(self.load_tokens) > 1:
            after = self.func.join(*self.load_tokens)
        self.func.store(value_wave, ptr_wave, after=after)

    def _emit_unmasked_store_of_masked_load_other(self, ptr: _Value, value: _Value) -> None:
        other_waves = value.other_waves or (value.other_wave, )
        if value.other_splat_const is not None and len(other_waves) != len(self._waves(ptr)):
            other_waves = tuple(
                self._materialize_splat_constant(value.other_splat_const) for _ in range(len(self._waves(ptr))))
        for mask_wave, ptr_wave, value_wave, other_wave, token in zip(value.mask_waves or (value.mask_wave, ),
                                                                      self._waves(ptr), self._waves(value), other_waves,
                                                                      value.tokens or (value.token, )):
            with self.func.where(mask_wave, [self.dsl.mem_token_type()]) as where_op:
                token = self.func.store(value_wave, ptr_wave, after=token)
                self.func.yield_([token])
            block = where_op.elseRegion.blocks.append()
            with self.dsl.InsertionPoint(block):
                token = self.func.store(other_wave, ptr_wave)
                self.dsl.wave.YieldOp([token])
        self.load_tokens.clear()

    def _load_other_value(self, other: _Value, elem_type: str) -> _Value:
        if other.wave is None:
            raise NotImplementedError("wave_amd masked tt.load `other` requires a Wave value")
        if other.waves:
            return other
        splat_const = None
        if other.splat_const is not None and other.splat_const[2] is None:
            splat_const = (other.splat_const[0], other.splat_const[1], None)
        return _Value(wave=other.wave, elem_type=other.elem_type or elem_type, splat_const=splat_const)

    def _materialize_splat_constant(self, constant: Tuple[object, str, Optional[int]]):
        value, elem_type, width = constant
        scalar = self.func.constant(self._scalar_type(elem_type), value)
        if width is None:
            return scalar
        return self.func.splat(scalar, self._scalar_type(elem_type), width)

    def _negative_one_like(self, value: _Value, index: int = 0):
        elem_type = value.elem_type or "i32"
        scalar = self.func.constant(self._scalar_type(elem_type), -1)
        wave = value.waves[index] if value.waves else value.wave
        if str(wave.type).startswith("!wave.simd<"):
            return self.func.splat(scalar, self._scalar_type(elem_type), self.width)
        return scalar

    def _select_operands(self, true_value: _Value, false_value: _Value):
        true_wave = true_value.wave
        false_wave = false_value.wave
        true_is_simd = str(true_wave.type).startswith("!wave.simd<")
        false_is_simd = str(false_wave.type).startswith("!wave.simd<")
        elem_type = true_value.elem_type or false_value.elem_type
        if true_is_simd and not false_is_simd:
            false_wave = self.func.splat(false_wave, self._scalar_type(false_value.elem_type or elem_type), self.width)
        elif false_is_simd and not true_is_simd:
            true_wave = self.func.splat(true_wave, self._scalar_type(true_value.elem_type or elem_type), self.width)
        return true_wave, false_wave

    def _memory_mask(self, op):
        name = op.get_name()
        if name == "tt.load" and op.get_num_operands() >= 2:
            return op.get_operand(1)
        if name == "tt.store" and op.get_num_operands() >= 3:
            return op.get_operand(2)
        return None

    def _index(self, state: _Value, index: int = 0):
        if state.index is not None and index == 0:
            return state.index
        if state.indexes:
            return state.indexes[index]
        expr, bindings = self._expr_and_bindings(state, index)
        if expr is None:
            raise NotImplementedError("wave_amd requires symbolic tt.addptr offsets")
        return self.func.index_expr(expr, bindings, self.dsl.simd_type(self.dsl.index_type(), self.width))

    def _indexes(self, state: _Value, layout: _BlockedLayout):
        if state.indexes:
            return state.indexes
        indexes = tuple(self._index(state, index) for index in range(layout.registers))
        state.indexes = indexes
        state.index = indexes[0]
        return indexes

    def _expr_and_bindings(self, state: _Value, index: int = 0):
        if state.exprs:
            return state.exprs[index], dict(state.bindings_by_wave[index])
        if state.expr is not None:
            return state.expr, dict(state.bindings)
        if state.const is None:
            return None, {}

        name = f"c{state.const}"
        sym = self._sym(name)
        return sym, {sym: state.wave}

    def _lane_id(self):
        if self._lane_value is None:
            self._lane_value = self.func.lane_id(self.dsl.i32(), self.width)
        return self._lane_value

    def _thread_id_and_sym(self):
        if self.num_warps == 1:
            return self._lane_id(), self._sym("lid")
        if self._workitem_value is None:
            self._workitem_value = self.func.workitem_id(axis=0, element_type=self.dsl.i32(), width=self.width)
        return self._workitem_value, self._sym("wi")

    def _splat_i32(self, value: int):
        scalar = self.func.constant(self.dsl.i32(), value)
        return self.func.splat(scalar, self.dsl.i32(), self.width)

    def _waves(self, state: _Value) -> Tuple[object, ...]:
        if state.waves:
            return state.waves
        if state.wave is None:
            return ()
        return (state.wave, )

    def _coerce_tensor(self, state: _Value, layout: _BlockedLayout, info: _TensorInfo) -> _Value:
        if state.layout is not None:
            if state.layout != layout:
                raise NotImplementedError("wave_amd elementwise operands must currently share one layout")
            return state
        if state.waves and len(state.waves) == layout.registers:
            state.layout = layout
            state.shape = info.shape
            return state
        if state.wave is None:
            raise NotImplementedError("wave_amd cannot broadcast an unsupported tensor value")
        waves = tuple(
            self.func.splat(state.wave, self._scalar_type(state.elem_type or info.elem_type), self.width)
            for _ in range(layout.registers))
        expr, bindings = self._expr_and_bindings(state)
        exprs = tuple(expr for _ in range(layout.registers))
        bindings_by_wave = tuple(dict(bindings) for _ in range(layout.registers))
        return _Value(
            wave=waves[0],
            waves=waves,
            elem_type=state.elem_type or info.elem_type,
            const=state.const,
            expr=expr,
            exprs=exprs,
            bindings=bindings,
            bindings_by_wave=bindings_by_wave,
            ptr_base=state.ptr_base,
            mask_id=state.mask_id,
            splat_const=state.splat_const,
            layout=layout,
            shape=info.shape,
        )

    def _coordinate_value(self, layout: _BlockedLayout, info: _TensorInfo, axis: int, start: int = 0) -> _Value:
        thread, sym = self._thread_id_and_sym()
        waves = []
        exprs = []
        bindings_by_wave = []
        for register in range(layout.registers):
            expr = layout.coord_expr(self.dsl, sym, register, axis)
            if start:
                expr = expr + start
            bindings = {sym: thread}
            if len(layout.shape) == 1:
                offset = register * self.width * self.num_warps + start
                wave = thread if offset == 0 else self.func.addi(thread, self._splat_i32(offset))
                waves.append(wave)
            exprs.append(expr)
            bindings_by_wave.append(bindings)
        return _Value(
            wave=waves[0] if waves else None,
            waves=tuple(waves),
            elem_type=info.elem_type,
            expr=exprs[0],
            exprs=tuple(exprs),
            bindings=bindings_by_wave[0],
            bindings_by_wave=tuple(bindings_by_wave),
            layout=layout,
            shape=info.shape,
            coord_axis=axis,
            coord_start=start,
        )

    def _result_tensor_info(self, op) -> Optional[_TensorInfo]:
        info = _wave_amd_native().get_result_tensor_info(op, 0)
        return _pack_tensor_info(info)

    def _layout_for_info(self, info: Optional[_TensorInfo]) -> Optional[_BlockedLayout]:
        if info is None:
            return None
        return _BlockedLayout.for_tensor(info, self.width, self.num_warps, self.num_ctas)

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


def _pack_tensor_info(info) -> Optional[_TensorInfo]:
    if info is None:
        return None
    shape, elem_type, is_pointer = info
    return _TensorInfo(tuple(int(dim) for dim in shape), str(elem_type), bool(is_pointer))


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

    required = ("get_program_id_axis", "get_cmpi_predicate", "get_arith_constant_splat", "get_result_tensor_info")
    missing = [name for name in required if not hasattr(wave_amd, name)]
    if missing:
        raise RuntimeError("wave_amd lowering requires the Triton wave_amd native extension to expose "
                           f"{', '.join(missing)}. Rebuild Triton with the updated wave_amd backend.")
    return wave_amd

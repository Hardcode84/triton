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


def _log2_int(value: int) -> int:
    if not _is_power_of_two(value):
        raise ValueError(f"expected a power of two, got {value}")
    return value.bit_length() - 1


@dataclass(frozen=True)
class _TensorInfo:
    shape: Tuple[int, ...]
    elem_type: str
    is_pointer: bool = False


@dataclass(frozen=True)
class _AffineIndex:
    const: int
    coeffs: Tuple[int, ...]


@dataclass(frozen=True)
class _SymbolicIndex:
    const: object
    coeffs: Tuple[object, ...]
    bindings: Dict[object, object]


@dataclass(frozen=True)
class _DotPointerLayout:
    shape: Tuple[int, ...]
    offset: _SymbolicIndex

    def expr(self, coords: Tuple[object, ...]):
        if len(coords) != len(self.offset.coeffs):
            raise ValueError("coordinate rank must match symbolic pointer rank")
        expr = self.offset.const
        for coord, coeff in zip(coords, self.offset.coeffs):
            expr = expr + coord * coeff
        return expr


@dataclass(frozen=True)
class _DotConfig:
    m: int
    n: int
    k: int
    m_tiles: int
    n_tiles: int
    k_steps: int


@dataclass(frozen=True)
class _DotFragmentGrid:
    role: int
    m_tiles: int
    n_tiles: int
    k_steps: int
    fragments: Tuple[object, ...]
    tokens: Tuple[object, ...] = ()

    def a(self, m_tile: int, k_step: int):
        return self.fragments[m_tile * self.k_steps + k_step]

    def b(self, n_tile: int, k_step: int):
        return self.fragments[n_tile * self.k_steps + k_step]

    def c(self, m_tile: int, n_tile: int):
        return self.fragments[m_tile * self.n_tiles + n_tile]


@dataclass(frozen=True)
class _BlockedLayout:
    shape: Tuple[int, ...]
    width: int
    num_warps: int
    num_ctas: int
    registers: int

    @classmethod
    def for_tensor(cls, info: _TensorInfo, width: int, num_warps: int, num_ctas: int) -> "_BlockedLayout":
        if num_warps < 1 or not _is_power_of_two(num_warps):
            raise NotImplementedError("wave_amd layout lowering requires power-of-two num_warps")
        if num_ctas < 1 or not _is_power_of_two(num_ctas):
            raise NotImplementedError("wave_amd layout lowering requires power-of-two num_ctas")
        elements = _product(info.shape)
        if any(not _is_power_of_two(dim) for dim in info.shape):
            raise NotImplementedError("wave_amd layout lowering requires power-of-two tensor dimensions")
        if not _is_power_of_two(elements):
            raise NotImplementedError("wave_amd layout lowering requires power-of-two tensor shapes")
        threads = width * num_warps
        cga_threads = threads * num_ctas
        if elements < cga_threads or elements % cga_threads != 0:
            raise NotImplementedError("wave_amd layout lowering requires tensor elements to be a multiple of "
                                      f"num_ctas * num_warps * warp_size, got {elements}")
        return cls(
            shape=info.shape,
            width=width,
            num_warps=num_warps,
            num_ctas=num_ctas,
            registers=elements // cga_threads,
        )

    @property
    def elements_per_cta(self) -> int:
        return _product(self.shape) // self.num_ctas

    @property
    def threads_per_cta(self) -> int:
        return self.width * self.num_warps

    def index_expr(self, dsl, lane_sym, cta_sym, register: int):
        if register < 0 or register >= self.registers:
            raise IndexError(register)
        flat = lane_sym
        if register != 0:
            flat = flat + register * self.threads_per_cta
        if self.num_ctas != 1:
            if cta_sym is None:
                raise ValueError("wave_amd multi-CTA layout requires a CTA symbol")
            flat = cta_sym * self.elements_per_cta + flat
        return flat

    def coord_expr(self, dsl, lane_sym, cta_sym, register: int, axis: int):
        if axis < 0 or axis >= len(self.shape):
            raise IndexError(axis)
        flat = self.index_expr(dsl, lane_sym, cta_sym, register)
        stride = _product(self.shape[axis + 1:])
        if stride != 1:
            flat = dsl.floor(flat / stride)
        dim = self.shape[axis]
        if dim == 1:
            return dsl.sym_ctx.int_(0)
        return dsl.mod(flat, dim)


@dataclass
class _LanePayload:
    wave: object = None
    waves: Tuple[object, ...] = field(default_factory=tuple)


@dataclass
class _DotPayload:
    fragment: object = None
    fragments: Tuple[object, ...] = field(default_factory=tuple)
    dot_grid: Optional[_DotFragmentGrid] = None


@dataclass
class _ConstantPayload:
    const: Optional[int] = None
    splat_const: Optional[Tuple[object, str, Optional[int]]] = None


@dataclass
class _SymbolicPayload:
    affine: Optional[_AffineIndex] = None
    expr: object = None
    exprs: Tuple[object, ...] = field(default_factory=tuple)
    bindings: Dict[object, object] = field(default_factory=dict)
    bindings_by_wave: Tuple[Dict[object, object], ...] = field(default_factory=tuple)
    sym_index: Optional[_SymbolicIndex] = None
    coord_axis: Optional[int] = None
    coord_start: int = 0


@dataclass
class _IndexPayload:
    index: object = None
    indexes: Tuple[object, ...] = field(default_factory=tuple)


@dataclass
class _MemoryPayload:
    token: object = None
    tokens: Tuple[object, ...] = field(default_factory=tuple)


@dataclass
class _PointerPayload:
    ptr_base: object = None
    ptr_offset_affine: Optional[_AffineIndex] = None
    ptr_dot_layout: Optional[_DotPointerLayout] = None


@dataclass
class _MaskPayload:
    mask_id: Optional[int] = None
    mask_wave: object = None
    mask_waves: Tuple[object, ...] = field(default_factory=tuple)
    mask_components: Tuple[object, ...] = field(default_factory=tuple)
    other_wave: object = None
    other_waves: Tuple[object, ...] = field(default_factory=tuple)
    other_splat_const: Optional[Tuple[object, str, Optional[int]]] = None
    cmpi_predicate: Optional[str] = None
    cmpi_lhs: Optional["_Value"] = None
    cmpi_rhs: Optional["_Value"] = None


@dataclass
class _Value:
    """Lowered SSA value grouped by concern rather than one flat union."""

    elem_type: Optional[str] = None
    layout: Optional[_BlockedLayout] = None
    shape: Optional[Tuple[int, ...]] = None
    lanes: _LanePayload = field(default_factory=_LanePayload)
    dot: _DotPayload = field(default_factory=_DotPayload)
    constant: _ConstantPayload = field(default_factory=_ConstantPayload)
    symbolic: _SymbolicPayload = field(default_factory=_SymbolicPayload)
    indexing: _IndexPayload = field(default_factory=_IndexPayload)
    memory: _MemoryPayload = field(default_factory=_MemoryPayload)
    pointer: _PointerPayload = field(default_factory=_PointerPayload)
    mask: _MaskPayload = field(default_factory=_MaskPayload)


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
        self._workgroup_values: Dict[int, object] = {}
        self.dot_operand_roles: Dict[int, int] = {}
        self.all_entry_ops: Tuple[object, ...] = ()
        self.ops_by_region: Dict[int, Tuple[object, ...]] = {}
        self.loop_yields: Dict[int, _Value] = {}

    def lower(self) -> Tuple[str, str]:
        name = self.module.get_entry_func_name()
        if not name:
            raise NotImplementedError("wave_amd expects a kernel tt.func entry point")

        ops = self._collect_entry_ops(name)
        func_op = self.module.get_function(name)
        self._collect_dot_operand_roles(self.all_entry_ops)
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
                    self.all_entry_ops = tuple(pending_body_ops)
                    self.ops_by_region = self._group_ops_by_parent_region(pending_body_ops)
                    entry_ops = list(self.ops_by_region.get(op.get_region(0).id(), ()))
                pending_body_ops.clear()
                return
            pending_body_ops.append(op)

        self.module.walk(visit)
        return entry_ops

    def _group_ops_by_parent_region(self, ops: Sequence[object]) -> Dict[int, Tuple[object, ...]]:
        grouped = {}
        for op in ops:
            block = op.get_block()
            if block is None:
                continue
            parent_region = block.get_parent()
            if parent_region is None:
                continue
            grouped.setdefault(parent_region.id(), []).append(op)
        return {region_id: tuple(region_ops) for region_id, region_ops in grouped.items()}

    def _region_ops(self, op, region_index: int = 0) -> Tuple[object, ...]:
        return self.ops_by_region.get(op.get_region(region_index).id(), ())

    def _collect_dot_operand_roles(self, ops: Sequence[object]) -> None:
        for op in ops:
            if op.get_name() != "tt.dot":
                continue
            for role in (0, 1):
                value_id = op.get_operand(role).id()
                existing = self.dot_operand_roles.get(value_id)
                if existing is not None and existing != role:
                    raise NotImplementedError("wave_amd tt.dot lowering cannot reuse one value as both dot operands")
                self.dot_operand_roles[value_id] = role

    def _bind_arguments(self, func_op, wave_args) -> None:
        signatures = self.module.get_function_signature(func_op)
        for index in range(func_op.get_num_args()):
            signature = signatures[index]
            state = _Value(elem_type=_signature_element_type(signature), lanes=_LanePayload(wave=wave_args[index]))
            if signature.startswith("*"):
                state.pointer.ptr_base = wave_args[index]
            else:
                sym = self._sym(f"arg_{index}")
                state.symbolic.expr = sym
                state.symbolic.bindings = {sym: wave_args[index]}
                state.symbolic.sym_index = _SymbolicIndex(sym, (), {sym: wave_args[index]})
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
            self._lower_binary(op, lambda lhs, rhs: lhs + rhs, self.func.addi, _affine_add, _symbolic_add)
        elif name == "arith.subi":
            self._lower_subi(op)
        elif name == "arith.muli":
            self._lower_binary(op, lambda lhs, rhs: lhs * rhs, self.func.muli, _affine_mul, _symbolic_mul)
        elif name == "arith.andi":
            self._lower_mask_binary(op, "and")
        elif name == "arith.ori":
            self._lower_mask_binary(op, "or")
        elif name == "arith.divsi":
            self._lower_scalar_index_divrem(op, "divsi")
        elif name == "arith.remsi":
            self._lower_scalar_index_divrem(op, "remsi")
        elif name == "arith.index_cast":
            self._lower_index_cast(op)
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
        elif name == "tt.dot":
            self._lower_dot(op)
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
        elif name == "scf.for":
            self._lower_for(op)
        elif name == "scf.yield":
            self._lower_yield(op)
        else:
            raise NotImplementedError(f"wave_amd TTIR lowering does not support op: {name}")

    def _lower_constant(self, op) -> None:
        constant = self._arith_constant_splat(op)
        if constant is None:
            value = op.get_constant_value()
            elem_type = _value_element_type_name(op.get_result(0).get_type())
            if value is not None and elem_type is not None:
                constant = (value, elem_type, None)
        state = _Value()
        if constant is not None:
            value, elem_type, width = constant
            scalar = self.func.constant(self._scalar_type(elem_type), value)
            state.elem_type = elem_type
            state.constant.const = value if isinstance(value, int) and width is None else None
            if isinstance(value, int):
                state.symbolic.affine = _AffineIndex(value, ())
                state.symbolic.sym_index = _SymbolicIndex(value, (), {})
            state.constant.splat_const = (value, elem_type, width)
            if width is None:
                state.lanes.wave = scalar
            else:
                info = self._result_tensor_info(op)
                if info is None:
                    raise NotImplementedError("wave_amd dense constants require tensor result type metadata")
                try:
                    layout = self._layout_for_info(info)
                except NotImplementedError:
                    if not self._can_defer_coordinate_layout(info):
                        raise
                    state.elem_type = elem_type
                    state.constant.const = value if isinstance(value, int) else None
                    state.symbolic.affine = _AffineIndex(value,
                                                         (0, ) * len(info.shape)) if isinstance(value, int) else None
                    state.symbolic.sym_index = _SymbolicIndex(value, (0, ) *
                                                              len(info.shape), {}) if isinstance(value, int) else None
                    state.constant.splat_const = (value, elem_type, width)
                    state.shape = info.shape
                    self._set_result(op, state)
                    return
                waves = tuple(
                    self.func.splat(scalar, self._scalar_type(elem_type), self.width) for _ in range(layout.registers))
                state.lanes.wave = waves[0]
                state.lanes.waves = waves
                state.layout = layout
                state.shape = info.shape
                if isinstance(value, int):
                    state.symbolic.affine = _AffineIndex(value, (0, ) * len(info.shape))
                    state.symbolic.sym_index = _SymbolicIndex(value, (0, ) * len(info.shape), {})
        self._set_result(op, state)

    def _lower_program_id(self, op) -> None:
        axis = self._program_id_axis(op)
        if axis is None:
            raise NotImplementedError(
                "wave_amd cannot lower tt.get_program_id until ProgramDimAttr is exposed structurally")
        pid = self._workgroup_id(axis)
        sym = self._sym(f"pid_{axis}")
        if axis == 0 and self.num_ctas > 1:
            wg_sym = self._workgroup_sym(axis)
            expr = self.dsl.floor(wg_sym / self.num_ctas)
            pid_index = self.func.index_expr(expr, {wg_sym: pid}, self.dsl.index_type())
            pid = self.func.index_cast(pid_index, self.dsl.i32())
            self._set_result(
                op,
                _Value(
                    elem_type="i32",
                    lanes=_LanePayload(wave=pid),
                    symbolic=_SymbolicPayload(expr=expr, bindings={wg_sym: self._workgroup_id(axis)},
                                              sym_index=_SymbolicIndex(expr, (), {wg_sym: self._workgroup_id(axis)})),
                ),
            )
            return
        self._set_result(
            op,
            _Value(
                elem_type="i32",
                lanes=_LanePayload(wave=pid),
                symbolic=_SymbolicPayload(expr=sym, bindings={sym: pid}, sym_index=_SymbolicIndex(sym, (), {sym: pid})),
            ),
        )

    def _lower_make_range(self, op) -> None:
        start = op.get_int_attr("start")
        end = op.get_int_attr("end")
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.make_range requires tensor result type metadata")
        if len(info.shape) != 1 or info.shape[0] != end - start:
            raise NotImplementedError("wave_amd tt.make_range shape must match end - start")

        try:
            layout = self._layout_for_info(info)
        except NotImplementedError:
            if len(info.shape) == 1 and self._can_defer_coordinate_layout(info):
                self._set_result(
                    op,
                    _Value(
                        elem_type="i32",
                        shape=info.shape,
                        symbolic=_SymbolicPayload(sym_index=_symbolic_coord(len(info.shape), 0, start), coord_axis=0,
                                                  coord_start=start),
                    ),
                )
                return
            raise
        state = self._coordinate_value(layout, info, axis=0, start=start)
        self._set_result(
            op,
            _Value(
                elem_type="i32",
                layout=layout,
                shape=info.shape,
                lanes=_LanePayload(wave=state.lanes.wave, waves=state.lanes.waves),
                symbolic=_SymbolicPayload(affine=state.symbolic.affine, expr=state.symbolic.expr,
                                          exprs=state.symbolic.exprs, bindings=state.symbolic.bindings,
                                          bindings_by_wave=state.symbolic.bindings_by_wave,
                                          sym_index=state.symbolic.sym_index, coord_axis=0, coord_start=start),
            ),
        )

    def _lower_expand_dims(self, op) -> None:
        src = self._value(op.get_operand(0))
        axis = op.get_int_attr("axis")
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.expand_dims requires tensor result type metadata")
        if src.symbolic.coord_axis is None:
            sym_index = _symbolic_expand_dims(src.symbolic.sym_index, axis, len(info.shape))
            if sym_index is None:
                raise NotImplementedError("wave_amd tt.expand_dims currently supports symbolic index tensors")
            try:
                layout = self._layout_for_info(info)
            except NotImplementedError:
                if self._can_defer_coordinate_layout(info):
                    self._set_result(
                        op,
                        _Value(
                            elem_type=src.elem_type or info.elem_type,
                            shape=info.shape,
                            symbolic=_SymbolicPayload(sym_index=sym_index),
                        ),
                    )
                    return
                raise
            self._set_result(op, self._symbolic_tensor_value(layout, info, sym_index, materialize_waves=False))
            return
        coord_axis = src.symbolic.coord_axis + 1 if axis <= src.symbolic.coord_axis else src.symbolic.coord_axis
        try:
            layout = self._layout_for_info(info)
        except NotImplementedError:
            if self._can_defer_coordinate_layout(info):
                self._set_result(
                    op,
                    _Value(
                        elem_type=src.elem_type or info.elem_type,
                        shape=info.shape,
                        symbolic=_SymbolicPayload(
                            sym_index=_symbolic_coord(len(info.shape), coord_axis, src.symbolic.coord_start),
                            coord_axis=coord_axis, coord_start=src.symbolic.coord_start),
                    ),
                )
                return
            raise
        state = self._coordinate_value(layout, _TensorInfo(info.shape, src.elem_type or info.elem_type), coord_axis,
                                       src.symbolic.coord_start)
        self._set_result(op, state)

    def _lower_broadcast(self, op) -> None:
        src = self._value(op.get_operand(0))
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.broadcast requires tensor result type metadata")
        layout = self._layout_for_info(info)
        if src.symbolic.coord_axis is not None:
            state = self._coordinate_value(layout, _TensorInfo(info.shape, src.elem_type or info.elem_type),
                                           src.symbolic.coord_axis, src.symbolic.coord_start)
        elif src.pointer.ptr_dot_layout is not None:
            state = _Value(
                elem_type=src.elem_type,
                shape=info.shape,
                pointer=_PointerPayload(
                    ptr_base=src.pointer.ptr_base, ptr_offset_affine=None, ptr_dot_layout=_DotPointerLayout(
                        info.shape, _symbolic_broadcast(src.pointer.ptr_dot_layout.offset, len(info.shape)))),
            )
        elif src.symbolic.sym_index is not None and (not src.lanes.waves or src.layout != layout):
            state = self._symbolic_tensor_value(layout, info,
                                                _symbolic_broadcast(src.symbolic.sym_index, len(info.shape)),
                                                materialize_waves=False)
        else:
            state = self._coerce_tensor(src, layout, info)
        if src.pointer.ptr_dot_layout is not None:
            state.pointer.ptr_base = src.pointer.ptr_base
            state.pointer.ptr_offset_affine = None
            state.pointer.ptr_dot_layout = _DotPointerLayout(
                info.shape, _symbolic_broadcast(src.pointer.ptr_dot_layout.offset, len(info.shape)))
        self._set_result(op, state)

    def _lower_splat(self, op) -> None:
        src = self._value(op.get_operand(0))
        info = self._result_tensor_info(op)
        layout = None
        if info is not None:
            try:
                layout = self._layout_for_info(info)
            except NotImplementedError:
                if not self._can_defer_coordinate_layout(info):
                    raise
                if src.pointer.ptr_base is not None:
                    self._set_result(
                        op,
                        _Value(
                            elem_type=src.elem_type,
                            shape=info.shape,
                            pointer=_PointerPayload(ptr_base=src.pointer.ptr_base),
                            mask=_MaskPayload(mask_id=src.mask.mask_id),
                        ),
                    )
                    return
                expr, bindings = self._expr_and_bindings(src)
                self._set_result(
                    op,
                    _Value(
                        elem_type=src.elem_type or info.elem_type,
                        shape=info.shape,
                        constant=_ConstantPayload(
                            const=src.constant.const,
                            splat_const=(src.constant.splat_const[0], src.constant.splat_const[1],
                                         _product(info.shape)) if src.constant.splat_const is not None else None),
                        symbolic=_SymbolicPayload(affine=_affine_splat(src.symbolic.affine,
                                                                       len(info.shape)), expr=expr, bindings=bindings,
                                                  sym_index=_symbolic_splat(src.symbolic.sym_index, len(info.shape))),
                        mask=_MaskPayload(mask_id=src.mask.mask_id),
                    ),
                )
                return
        if src.pointer.ptr_base is not None:
            state = _Value(
                elem_type=src.elem_type,
                layout=layout,
                shape=None if info is None else info.shape,
                lanes=_LanePayload(wave=src.pointer.ptr_base),
                pointer=_PointerPayload(ptr_base=src.pointer.ptr_base),
                mask=_MaskPayload(mask_id=src.mask.mask_id),
            )
        else:
            expr, bindings = self._expr_and_bindings(src)
            splat_const = None
            if src.constant.splat_const is not None and src.constant.splat_const[2] is None and layout is not None:
                splat_const = (src.constant.splat_const[0], src.constant.splat_const[1], _product(info.shape))
            affine = _affine_splat(src.symbolic.affine, len(info.shape)) if layout is not None else src.symbolic.affine
            sym_index = _symbolic_splat(src.symbolic.sym_index, len(
                info.shape)) if layout is not None else src.symbolic.sym_index
            if layout is not None:
                waves = tuple(
                    self.func.splat(src.lanes.wave, self._scalar_type(src.elem_type), self.width)
                    for _ in range(layout.registers))
                exprs = tuple(expr for _ in range(layout.registers))
                bindings_by_wave = tuple(dict(bindings) for _ in range(layout.registers))
                state = _Value(
                    elem_type=src.elem_type,
                    layout=layout,
                    shape=info.shape,
                    lanes=_LanePayload(wave=waves[0], waves=waves),
                    constant=_ConstantPayload(const=src.constant.const, splat_const=splat_const),
                    symbolic=_SymbolicPayload(affine=affine, expr=expr, exprs=exprs, bindings=bindings,
                                              bindings_by_wave=bindings_by_wave, sym_index=sym_index),
                    mask=_MaskPayload(mask_id=src.mask.mask_id),
                )
            else:
                wave = self.func.splat(src.lanes.wave, self._scalar_type(src.elem_type), self.width)
                state = _Value(
                    elem_type=src.elem_type,
                    lanes=_LanePayload(wave=wave, waves=(wave, )),
                    constant=_ConstantPayload(const=src.constant.const, splat_const=splat_const),
                    symbolic=_SymbolicPayload(affine=affine, expr=expr, exprs=(expr, ), bindings=bindings,
                                              bindings_by_wave=(bindings, ), sym_index=sym_index),
                    mask=_MaskPayload(mask_id=src.mask.mask_id),
                )
        self._set_result(op, state)

    def _lower_binary(self, op, expr_builder, wave_builder, affine_builder=None, symbolic_builder=None) -> None:
        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        mask_id = _merge_mask_ids(lhs, rhs)
        info = self._result_tensor_info(op)
        if info is not None:
            try:
                layout = self._layout_for_info(info)
            except NotImplementedError:
                if not self._can_defer_coordinate_layout(info) or symbolic_builder is None:
                    raise
                self._set_result(
                    op,
                    _Value(
                        elem_type=lhs.elem_type or rhs.elem_type,
                        shape=info.shape,
                        symbolic=_SymbolicPayload(
                            affine=affine_builder(lhs.symbolic.affine, rhs.symbolic.affine)
                            if affine_builder is not None else None,
                            sym_index=symbolic_builder(lhs.symbolic.sym_index, rhs.symbolic.sym_index)),
                        mask=_MaskPayload(mask_id=mask_id),
                    ),
                )
                return
            lhs = self._coerce_tensor(lhs, layout, info)
            rhs = self._coerce_tensor(rhs, layout, info)
            waves = ()
            if self._can_emit_binary_waves(lhs, rhs):
                waves = tuple(
                    wave_builder(lhs_wave, rhs_wave) for lhs_wave, rhs_wave in zip(lhs.lanes.waves, rhs.lanes.waves))
            affine = affine_builder(lhs.symbolic.affine, rhs.symbolic.affine) if affine_builder is not None else None
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
            sym_index = symbolic_builder(lhs.symbolic.sym_index,
                                         rhs.symbolic.sym_index) if symbolic_builder is not None else None
            self._set_result(
                op,
                _Value(
                    elem_type=lhs.elem_type or rhs.elem_type,
                    layout=layout,
                    shape=info.shape,
                    lanes=_LanePayload(wave=waves[0] if waves else None, waves=waves),
                    symbolic=_SymbolicPayload(affine=affine, expr=exprs[0], exprs=tuple(exprs),
                                              bindings=bindings_by_wave[0], bindings_by_wave=tuple(bindings_by_wave),
                                              sym_index=sym_index),
                    mask=_MaskPayload(mask_id=mask_id),
                ),
            )
            return

        wave = wave_builder(lhs.lanes.wave, rhs.lanes.wave)
        expr = None
        bindings = {}
        if expr_builder is not None:
            lhs_expr, lhs_bindings = self._expr_and_bindings(lhs)
            rhs_expr, rhs_bindings = self._expr_and_bindings(rhs)
            if lhs_expr is not None and rhs_expr is not None:
                expr = expr_builder(lhs_expr, rhs_expr)
                bindings = {**lhs_bindings, **rhs_bindings}
        affine = affine_builder(lhs.symbolic.affine, rhs.symbolic.affine) if affine_builder is not None else None
        sym_index = symbolic_builder(lhs.symbolic.sym_index,
                                     rhs.symbolic.sym_index) if symbolic_builder is not None else None
        self._set_result(
            op,
            _Value(
                elem_type=lhs.elem_type or rhs.elem_type,
                lanes=_LanePayload(wave=wave),
                symbolic=_SymbolicPayload(affine=affine, expr=expr, bindings=bindings, sym_index=sym_index),
                mask=_MaskPayload(mask_id=mask_id),
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
            affine = _affine_sub(lhs.symbolic.affine, rhs.symbolic.affine)
            waves = []
            exprs = []
            bindings_by_wave = []
            for index in range(layout.registers):
                if self._can_emit_binary_waves(lhs, rhs):
                    lhs_wave = lhs.lanes.waves[index]
                    rhs_wave = rhs.lanes.waves[index]
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
                    elem_type=lhs.elem_type or rhs.elem_type,
                    layout=layout,
                    shape=info.shape,
                    lanes=_LanePayload(wave=waves[0] if waves else None, waves=tuple(waves)),
                    symbolic=_SymbolicPayload(affine=affine, expr=exprs[0], exprs=tuple(exprs),
                                              bindings=bindings_by_wave[0], bindings_by_wave=tuple(bindings_by_wave),
                                              sym_index=_symbolic_sub(lhs.symbolic.sym_index, rhs.symbolic.sym_index)),
                    mask=_MaskPayload(mask_id=mask_id),
                ),
            )
            return

        neg_rhs = self.func.muli(rhs.lanes.wave, self._negative_one_like(rhs))
        wave = self.func.addi(lhs.lanes.wave, neg_rhs)
        expr = None
        bindings = {}
        lhs_expr, lhs_bindings = self._expr_and_bindings(lhs)
        rhs_expr, rhs_bindings = self._expr_and_bindings(rhs)
        if lhs_expr is not None and rhs_expr is not None:
            expr = lhs_expr - rhs_expr
            bindings = {**lhs_bindings, **rhs_bindings}
        affine = _affine_sub(lhs.symbolic.affine, rhs.symbolic.affine)
        self._set_result(
            op,
            _Value(
                elem_type=lhs.elem_type or rhs.elem_type,
                lanes=_LanePayload(wave=wave),
                symbolic=_SymbolicPayload(affine=affine, expr=expr, bindings=bindings,
                                          sym_index=_symbolic_sub(lhs.symbolic.sym_index, rhs.symbolic.sym_index)),
                mask=_MaskPayload(mask_id=mask_id),
            ),
        )

    def _lower_scalar_index_divrem(self, op, kind: str) -> None:
        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        if self._result_tensor_info(op) is not None:
            raise NotImplementedError(f"wave_amd {kind} currently supports only scalar integer values")
        if rhs.constant.const is None or rhs.constant.const <= 0:
            raise NotImplementedError(f"wave_amd {kind} currently requires a positive constant divisor")
        lhs_expr, lhs_bindings = self._expr_and_bindings(lhs)
        if lhs_expr is None:
            raise NotImplementedError(f"wave_amd {kind} currently requires a symbolic dividend")
        if kind == "divsi":
            expr = self.dsl.floor(lhs_expr / rhs.constant.const)
        elif kind == "remsi":
            expr = self.dsl.mod(lhs_expr, rhs.constant.const)
        else:
            raise AssertionError(kind)
        index = self.func.index_expr(expr, lhs_bindings, self.dsl.index_type())
        value = self.func.index_cast(index, self.dsl.i32())
        self._set_result(
            op,
            _Value(
                elem_type=lhs.elem_type or "i32",
                lanes=_LanePayload(wave=value),
                symbolic=_SymbolicPayload(expr=expr, bindings=lhs_bindings,
                                          sym_index=_SymbolicIndex(expr, (), dict(lhs_bindings))),
            ),
        )

    def _lower_index_cast(self, op) -> None:
        src = self._value(op.get_operand(0))
        elem_type = _value_element_type_name(op.get_result(0).get_type())
        if elem_type is None:
            raise NotImplementedError("wave_amd arith.index_cast requires a supported scalar result type")
        wave = self.func.index_cast(src.lanes.wave, self._scalar_type(elem_type))
        self._set_result(
            op,
            _Value(
                elem_type=elem_type,
                lanes=_LanePayload(wave=wave),
                constant=_ConstantPayload(const=src.constant.const),
                symbolic=_SymbolicPayload(affine=src.symbolic.affine, expr=src.symbolic.expr,
                                          bindings=src.symbolic.bindings, sym_index=src.symbolic.sym_index),
            ),
        )

    def _lower_mask_binary(self, op, kind: str) -> None:
        if kind != "and":
            raise NotImplementedError("wave_amd matmul mask lowering currently supports arith.andi only")
        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        info = self._result_tensor_info(op)
        if info is None or info.elem_type != "i1":
            raise NotImplementedError("wave_amd arith.andi currently supports tensor mask operands only")
        self._set_result(
            op,
            _Value(
                elem_type="i1",
                layout=lhs.layout or rhs.layout,
                shape=info.shape,
                mask=_MaskPayload(mask_components=(lhs, rhs)),
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
            compare_as_index = ((lhs.symbolic.sym_index is not None and not lhs.lanes.waves)
                                or (rhs.symbolic.sym_index is not None and not rhs.lanes.waves))
            if compare_as_index:
                index_info = _TensorInfo(info.shape, "index")
                if lhs.symbolic.sym_index is None or rhs.symbolic.sym_index is None:
                    raise NotImplementedError("wave_amd symbolic comparisons require symbolic operands")
                lhs = self._symbolic_tensor_value(layout, index_info, lhs.symbolic.sym_index)
                rhs = self._symbolic_tensor_value(layout, index_info, rhs.symbolic.sym_index)
            else:
                lhs_info = _TensorInfo(info.shape, lhs.elem_type or "i32")
                rhs_info = _TensorInfo(info.shape, rhs.elem_type or "i32")
                lhs = self._coerce_tensor(lhs, layout, lhs_info)
                rhs = self._coerce_tensor(rhs, layout, rhs_info)
            waves = tuple(
                self.func.cmpi(predicate, lhs_wave, rhs_wave)
                for lhs_wave, rhs_wave in zip(lhs.lanes.waves, rhs.lanes.waves))
            self._set_result(
                op,
                _Value(
                    elem_type="i1",
                    layout=layout,
                    shape=info.shape,
                    lanes=_LanePayload(wave=waves[0], waves=waves),
                    mask=_MaskPayload(cmpi_predicate=predicate, cmpi_lhs=lhs, cmpi_rhs=rhs),
                ),
            )
            return
        self._set_result(
            op,
            _Value(
                elem_type="i1",
                lanes=_LanePayload(wave=self.func.cmpi(predicate, lhs.lanes.wave, rhs.lanes.wave)),
                mask=_MaskPayload(cmpi_predicate=predicate, cmpi_lhs=lhs, cmpi_rhs=rhs),
            ),
        )

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
                self.func.select(cond_wave, true_wave, false_wave) for cond_wave, true_wave, false_wave in zip(
                    condition.lanes.waves, true_value.lanes.waves, false_value.lanes.waves))
            self._set_result(
                op,
                _Value(
                    elem_type=true_value.elem_type or false_value.elem_type,
                    layout=layout,
                    shape=info.shape,
                    lanes=_LanePayload(wave=waves[0], waves=waves),
                    mask=_MaskPayload(mask_id=_merge_mask_ids(true_value, false_value)),
                ),
            )
            return
        true_wave, false_wave = self._select_operands(true_value, false_value)
        self._set_result(
            op,
            _Value(
                elem_type=true_value.elem_type or false_value.elem_type,
                lanes=_LanePayload(wave=self.func.select(condition.lanes.wave, true_wave, false_wave)),
                mask=_MaskPayload(mask_id=_merge_mask_ids(true_value, false_value)),
            ),
        )

    def _lower_addptr(self, op) -> None:
        ptr = self._value(op.get_operand(0))
        offset = self._value(op.get_operand(1))
        info = self._result_tensor_info(op)
        if info is None:
            raise NotImplementedError("wave_amd tt.addptr requires tensor result type metadata")
        try:
            layout = self._layout_for_info(info)
        except NotImplementedError:
            if not self._can_defer_coordinate_layout(info):
                raise
            ptr_dot_layout = self._combined_pointer_layout(ptr, offset, info.shape)
            if ptr.pointer.ptr_base is None or ptr_dot_layout is None:
                raise NotImplementedError("wave_amd requires symbolic tt.addptr offsets")
            self._set_result(
                op,
                _Value(
                    elem_type=ptr.elem_type,
                    shape=info.shape,
                    pointer=_PointerPayload(ptr_base=ptr.pointer.ptr_base, ptr_offset_affine=offset.symbolic.affine,
                                            ptr_dot_layout=ptr_dot_layout),
                ),
            )
            return
        offset = self._coerce_tensor(offset, layout, _TensorInfo(info.shape, "i32"))
        indexes = self._indexes(offset, layout)
        base_waves = ptr.lanes.waves if ptr.lanes.waves else tuple(ptr.lanes.wave for _ in range(layout.registers))
        if ptr.pointer.ptr_base is not None:
            base_waves = tuple(ptr.pointer.ptr_base for _ in range(layout.registers))
        if len(base_waves) != layout.registers:
            raise NotImplementedError("wave_amd tt.addptr pointer and offset layouts must match")
        waves = tuple(self.func.ptr_add(base, index) for base, index in zip(base_waves, indexes))
        ptr_dot_layout = self._combined_pointer_layout(ptr, offset, info.shape)
        self._set_result(
            op,
            _Value(
                elem_type=ptr.elem_type,
                layout=layout,
                shape=info.shape,
                lanes=_LanePayload(wave=waves[0], waves=waves),
                pointer=_PointerPayload(ptr_base=ptr.pointer.ptr_base, ptr_offset_affine=offset.symbolic.affine,
                                        ptr_dot_layout=ptr_dot_layout),
            ),
        )

    def _combined_pointer_layout(self, ptr: _Value, offset: _Value, shape: Tuple[int,
                                                                                 ...]) -> Optional[_DotPointerLayout]:
        offset_sym = offset.symbolic.sym_index
        if ptr.pointer.ptr_dot_layout is not None:
            base_sym = _symbolic_broadcast(ptr.pointer.ptr_dot_layout.offset, len(shape))
            offset_sym = _symbolic_add(base_sym, offset_sym) if offset_sym is not None else base_sym
        if offset_sym is None:
            return None
        return _DotPointerLayout(shape, offset_sym)

    def _lower_load(self, op) -> None:
        role = self.dot_operand_roles.get(op.get_result(0).id())
        if role is not None:
            ptr = self._value(op.get_operand(0))
            info = self._result_tensor_info(op)
            mask = self._memory_mask(op)
            mask_value = None
            if mask is not None:
                if op.get_num_operands() != 3:
                    raise NotImplementedError("wave_amd masked tt.dot operand loads require an explicit zero `other`")
                self._expect_zero_dot_load_other(self._value(op.get_operand(2)), ptr.elem_type)
                if info is None or info.shape != (16, 16):
                    raise NotImplementedError("wave_amd masked tt.dot operand loads currently support one 16x16 tile")
                mask_value = self._value(mask)
                if not mask_value.mask.mask_components:
                    layout = ptr.layout or self._layout_for_info(info)
                    mask_value = self._coerce_tensor(mask_value, layout, _TensorInfo(info.shape, "i1"))
            grid = self._emit_dot_fragment_load(ptr, info, role, mask_value)
            self.load_tokens.extend(grid.tokens)
            self._set_result(
                op,
                _Value(
                    elem_type=info.elem_type,
                    shape=info.shape,
                    dot=_DotPayload(fragment=grid.fragments[0], fragments=grid.fragments, dot_grid=grid),
                    memory=_MemoryPayload(token=grid.tokens[0], tokens=grid.tokens),
                ),
            )
            return

        mask = self._memory_mask(op)
        if mask is not None:
            if op.get_num_operands() not in {2, 3}:
                raise NotImplementedError("wave_amd masked tt.load supports pointer, mask, and optional other only")
            ptr = self._value(op.get_operand(0))
            mask_value = self._value(mask)
            if mask_value.lanes.wave is None and not mask_value.lanes.waves:
                raise NotImplementedError("wave_amd masked load requires a Wave mask value")
            layout = ptr.layout or self._layout_for_info(self._result_tensor_info(op))
            mask_value = self._coerce_tensor(mask_value, layout, _TensorInfo(ptr.shape or layout.shape, "i1"))
            other = None
            if op.get_num_operands() == 3:
                other = self._load_other_value(self._value(op.get_operand(2)), ptr.elem_type)
                other = self._coerce_tensor(other, layout, _TensorInfo(ptr.shape or layout.shape, ptr.elem_type))
            values = []
            tokens = []
            for index, (ptr_wave, mask_wave) in enumerate(zip(self._waves(ptr), mask_value.lanes.waves)):
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
                        elem_type=ptr.elem_type,
                        layout=layout,
                        shape=ptr.shape,
                        lanes=_LanePayload(wave=values[0], waves=tuple(values)),
                        memory=_MemoryPayload(token=tokens[0], tokens=tuple(tokens)),
                        mask=_MaskPayload(mask_id=mask.id(), mask_wave=mask_value.lanes.waves[0],
                                          mask_waves=mask_value.lanes.waves, other_wave=other.lanes.waves[0],
                                          other_waves=other.lanes.waves, other_splat_const=other.constant.splat_const),
                    ),
                )
            else:
                self.load_tokens.extend(tokens)
                self._set_result(
                    op,
                    _Value(
                        elem_type=ptr.elem_type,
                        layout=layout,
                        shape=ptr.shape,
                        lanes=_LanePayload(wave=values[0], waves=tuple(values)),
                        memory=_MemoryPayload(token=tokens[0], tokens=tuple(tokens)),
                        mask=_MaskPayload(mask_id=mask.id()),
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
                elem_type=ptr.elem_type,
                layout=ptr.layout,
                shape=ptr.shape,
                lanes=_LanePayload(wave=values[0], waves=tuple(values)),
                memory=_MemoryPayload(token=tokens[0], tokens=tuple(tokens)),
            ),
        )

    def _lower_store(self, op) -> None:
        ptr = self._value(op.get_operand(0))
        value = self._value(op.get_operand(1))
        mask = self._memory_mask(op)
        if value.dot.fragment is not None:
            mask_value = None
            if mask is not None:
                if value.shape != (16, 16):
                    raise NotImplementedError("wave_amd masked tt.dot fragment stores currently support one 16x16 tile")
                mask_value = self._value(mask)
                if mask_value.lanes.wave is None and not mask_value.lanes.waves and not mask_value.mask.mask_components:
                    raise NotImplementedError("wave_amd masked tt.dot fragment stores require a Wave mask")
                if not mask_value.mask.mask_components:
                    layout = ptr.layout or self._layout_for_info(_TensorInfo(value.shape, "i1"))
                    mask_value = self._coerce_tensor(mask_value, layout, _TensorInfo(value.shape, "i1"))
            self._emit_fragment_store(ptr, value, mask_value)
            return
        if mask is None and value.mask.other_wave is not None and value.mask.mask_wave is not None:
            self._emit_unmasked_store_of_masked_load_other(ptr, value)
            return
        if mask is not None:
            if op.get_num_operands() != 3:
                raise NotImplementedError("wave_amd masked tt.store supports pointer, value, and mask operands only")
            if value.mask.mask_id is not None and value.mask.mask_id != mask.id():
                raise NotImplementedError(
                    "wave_amd masked load values must be stored under the same SSA mask that produced them")
            mask_value = self._value(mask)
            if mask_value.lanes.wave is None and not mask_value.lanes.waves:
                raise NotImplementedError("wave_amd masked store requires a Wave mask value")
            layout = ptr.layout or value.layout
            mask_value = self._coerce_tensor(mask_value, layout, _TensorInfo(ptr.shape or value.shape, "i1"))
            value = self._coerce_tensor(value, layout, _TensorInfo(ptr.shape or value.shape, value.elem_type))
            for mask_wave, ptr_wave, value_wave in zip(mask_value.lanes.waves, self._waves(ptr), value.lanes.waves):
                with self.func.where(mask_wave):
                    self._emit_store_wave(ptr_wave, value_wave)
            self.load_tokens.clear()
            return

        if value.mask.mask_id is not None:
            raise NotImplementedError("wave_amd masked load value cannot be stored without its producing SSA mask")
        self._emit_store(ptr, value)

    def _lower_for(self, op) -> None:
        if op.get_num_results() != 1 or op.get_num_operands() != 4:
            raise NotImplementedError("wave_amd scf.for K loops currently support one accumulator iter_arg")
        result_info = self._result_tensor_info(op)
        if result_info is None:
            raise NotImplementedError("wave_amd scf.for K loops require ranked tensor result metadata")

        lower = self._value(op.get_operand(0))
        upper = self._value(op.get_operand(1))
        step = self._value(op.get_operand(2))
        init = self._value(op.get_operand(3))
        init_grid = self._dot_accumulator_grid(init, result_info)
        body_ops = self._region_ops(op)
        if not body_ops:
            raise NotImplementedError("wave_amd scf.for K loops require a non-empty body")
        body_block = body_ops[0].get_block()
        if body_block.get_num_arguments() != 2:
            raise NotImplementedError("wave_amd scf.for K loops currently support one block iter_arg")

        saved_load_tokens = self.load_tokens
        self.load_tokens = []
        region_id = op.get_region(0).id()
        with self.func.for_loop(lower.lanes.wave, upper.lanes.wave, step.lanes.wave, init_args=init_grid.fragments,
                                nonzero_trip=True) as forop:
            iv_sym = self._sym(f"iv_{region_id}")
            self.values[body_block.get_argument(0).id()] = _Value(
                elem_type="index",
                lanes=_LanePayload(wave=forop.induction_variable),
                symbolic=_SymbolicPayload(expr=iv_sym, bindings={iv_sym: forop.induction_variable},
                                          sym_index=_SymbolicIndex(iv_sym, (), {iv_sym: forop.induction_variable})),
            )
            carry_fragments = tuple(forop.inner_iter_args)
            carry_grid = _DotFragmentGrid(
                role=2,
                m_tiles=init_grid.m_tiles,
                n_tiles=init_grid.n_tiles,
                k_steps=0,
                fragments=carry_fragments,
            )
            self.values[body_block.get_argument(1).id()] = _Value(
                elem_type=result_info.elem_type,
                shape=result_info.shape,
                dot=_DotPayload(fragment=carry_fragments[0], fragments=carry_fragments, dot_grid=carry_grid),
            )
            self._lower_ops(body_ops)
            yielded = self.loop_yields.pop(region_id, None)
            if yielded is None or yielded.dot.dot_grid is None:
                raise NotImplementedError("wave_amd scf.for K loops must yield a dot accumulator grid")
            self.func.yield_(yielded.dot.dot_grid.fragments)
        self.load_tokens = saved_load_tokens

        result_fragments = tuple(forop.results)
        result_grid = _DotFragmentGrid(
            role=2,
            m_tiles=init_grid.m_tiles,
            n_tiles=init_grid.n_tiles,
            k_steps=0,
            fragments=result_fragments,
        )
        self._set_result(
            op,
            _Value(
                elem_type=result_info.elem_type,
                shape=result_info.shape,
                dot=_DotPayload(fragment=result_fragments[0], fragments=result_fragments, dot_grid=result_grid),
            ),
        )

    def _lower_yield(self, op) -> None:
        if op.get_num_operands() != 1:
            raise NotImplementedError("wave_amd scf.for K loops currently support one yielded accumulator")
        block = op.get_block()
        if block is None or block.get_parent() is None:
            raise NotImplementedError("wave_amd scf.yield must be inside a loop region")
        self.loop_yields[block.get_parent().id()] = self._value(op.get_operand(0))

    def _lower_dot(self, op) -> None:
        lhs_info = self._value_tensor_info(op.get_operand(0))
        rhs_info = self._value_tensor_info(op.get_operand(1))
        acc_info = self._value_tensor_info(op.get_operand(2))
        result_info = self._result_tensor_info(op)
        config = self._dot_config(lhs_info, rhs_info, result_info)
        if acc_info != result_info:
            raise NotImplementedError("wave_amd tt.dot accumulator must match the dot result type")

        lhs = self._value(op.get_operand(0))
        rhs = self._value(op.get_operand(1))
        acc = self._value(op.get_operand(2))
        lhs_grid = self._dot_operand_grid(lhs, role=0, config=config)
        rhs_grid = self._dot_operand_grid(rhs, role=1, config=config)
        if lhs_grid is None or rhs_grid is None:
            raise NotImplementedError("wave_amd tt.dot currently requires operands produced by supported tt.load ops")
        result_fragments = []
        for m_tile in range(config.m_tiles):
            for n_tile in range(config.n_tiles):
                result = self._dot_accumulator_fragment(acc, result_info, m_tile, n_tile)
                for k_step in range(config.k_steps):
                    result = self.func.mma("wmma.f32.16x16x16.f16", lhs_grid.a(m_tile, k_step),
                                           rhs_grid.b(n_tile, k_step), result)
                result_fragments.append(result)
        result_grid = _DotFragmentGrid(
            role=2,
            m_tiles=config.m_tiles,
            n_tiles=config.n_tiles,
            k_steps=0,
            fragments=tuple(result_fragments),
        )
        self._set_result(
            op,
            _Value(
                elem_type=result_info.elem_type,
                shape=result_info.shape,
                dot=_DotPayload(fragment=result_fragments[0], fragments=tuple(result_fragments), dot_grid=result_grid),
            ),
        )

    def _dot_config(self, lhs: Optional[_TensorInfo], rhs: Optional[_TensorInfo],
                    result: Optional[_TensorInfo]) -> _DotConfig:
        if lhs is None or rhs is None or result is None:
            raise NotImplementedError("wave_amd tt.dot requires ranked tensor operands and result")
        if lhs.elem_type != "f16" or rhs.elem_type != "f16" or result.elem_type != "f32":
            raise NotImplementedError("wave_amd tt.dot currently supports f16 x f16 -> f32")
        if len(lhs.shape) != 2 or len(rhs.shape) != 2 or len(result.shape) != 2:
            raise NotImplementedError("wave_amd tt.dot currently supports rank-2 matrix operands")
        m, k = lhs.shape
        rhs_k, n = rhs.shape
        if rhs_k != k or result.shape != (m, n):
            raise NotImplementedError("wave_amd tt.dot found incompatible matrix shapes")
        if m < 16 or m % 16 != 0:
            raise NotImplementedError("wave_amd tt.dot M dimension must be a positive multiple of 16")
        if n < 16 or n % 16 != 0:
            raise NotImplementedError("wave_amd tt.dot N dimension must be a positive multiple of 16")
        if k < 16 or k % 16 != 0:
            raise NotImplementedError("wave_amd tt.dot K dimension must be a positive multiple of 16")
        m_tiles = m // 16
        n_tiles = n // 16
        output_tiles = m_tiles * n_tiles
        workers = self.num_warps * self.num_ctas
        if output_tiles % workers != 0:
            raise NotImplementedError(
                "wave_amd tt.dot requires output 16x16 tile count to be divisible by num_warps * num_ctas")
        return _DotConfig(m=m, n=n, k=k, m_tiles=m_tiles, n_tiles=n_tiles, k_steps=k // 16)

    def _dot_operand_grid(self, value: _Value, role: int, config: _DotConfig) -> Optional[_DotFragmentGrid]:
        if value.dot.dot_grid is None:
            if value.dot.fragments:
                k_steps = config.k_steps
                if role == 0 and config.m_tiles == 1 and len(value.dot.fragments) == k_steps:
                    return _DotFragmentGrid(role=role, m_tiles=1, n_tiles=0, k_steps=k_steps,
                                            fragments=value.dot.fragments)
                if role == 1 and config.n_tiles == 1 and len(value.dot.fragments) == k_steps:
                    return _DotFragmentGrid(role=role, m_tiles=0, n_tiles=1, k_steps=k_steps,
                                            fragments=value.dot.fragments)
            if value.dot.fragment is not None and config.k_steps == 1:
                return _DotFragmentGrid(role=role, m_tiles=1 if role == 0 else 0, n_tiles=1 if role == 1 else 0,
                                        k_steps=1, fragments=(value.dot.fragment, ))
            return None
        grid = value.dot.dot_grid
        if grid.role != role:
            raise NotImplementedError("wave_amd tt.dot operand fragment role does not match the dot operand")
        expected_outer_tiles = config.m_tiles if role == 0 else config.n_tiles
        actual_outer_tiles = grid.m_tiles if role == 0 else grid.n_tiles
        if actual_outer_tiles != expected_outer_tiles or grid.k_steps != config.k_steps:
            raise NotImplementedError("wave_amd tt.dot operand fragment grid must match the dot tile decomposition")
        return grid

    def _dot_accumulator_fragment(self, acc: _Value, result_info: _TensorInfo, m_tile: int, n_tile: int):
        frag_type = self._dot_fragment_type(role=2, elem_type=result_info.elem_type)
        if acc.dot.dot_grid is not None:
            return acc.dot.dot_grid.c(m_tile, n_tile)
        if acc.dot.fragment is not None and result_info.shape == (16, 16):
            return acc.dot.fragment
        if acc.constant.splat_const is not None:
            value, elem_type, width = acc.constant.splat_const
            if elem_type == result_info.elem_type and width == _product(result_info.shape) and value == 0:
                return self.func.fragment_fill(self.func.constant(self.dsl.i32(), 0), frag_type)
        raise NotImplementedError("wave_amd tt.dot currently supports only zero-splat f32 accumulators")

    def _dot_accumulator_grid(self, acc: _Value, result_info: _TensorInfo) -> _DotFragmentGrid:
        config = self._dot_config(_TensorInfo((result_info.shape[0], 16), "f16"),
                                  _TensorInfo((16, result_info.shape[1]), "f16"), result_info)
        if acc.dot.dot_grid is not None:
            if acc.dot.dot_grid.role != 2 or acc.dot.dot_grid.m_tiles != config.m_tiles or acc.dot.dot_grid.n_tiles != config.n_tiles:
                raise NotImplementedError("wave_amd scf.for accumulator grid must match the dot result shape")
            return acc.dot.dot_grid
        fragments = []
        for m_tile in range(config.m_tiles):
            for n_tile in range(config.n_tiles):
                fragments.append(self._dot_accumulator_fragment(acc, result_info, m_tile, n_tile))
        return _DotFragmentGrid(
            role=2,
            m_tiles=config.m_tiles,
            n_tiles=config.n_tiles,
            k_steps=0,
            fragments=tuple(fragments),
        )

    def _emit_dot_fragment_load(self, ptr: _Value, info: Optional[_TensorInfo], role: int,
                                mask: Optional[_Value] = None):
        if info is None:
            raise NotImplementedError("wave_amd tt.dot operand load requires ranked tensor result metadata")
        if info.elem_type != "f16" or len(info.shape) != 2:
            raise NotImplementedError("wave_amd tt.dot operand loads currently support rank-2 f16 tensors")
        if ptr.pointer.ptr_base is None:
            raise NotImplementedError("wave_amd tt.dot operand loads require a splatted pointer base")
        dot_layout = ptr.pointer.ptr_dot_layout
        fixed_affine = None
        if role == 0:
            m, k = info.shape
            if m < 16 or m % 16 != 0:
                raise NotImplementedError("wave_amd tt.dot A operand M dimension must be a positive multiple of 16")
            fixed_affine = _AffineIndex(0, (k, 1))
            if ptr.pointer.ptr_offset_affine != fixed_affine and dot_layout is None:
                self._expect_dot_pointer_layout(ptr, info.shape, fixed_affine, f"A operand row-major {m}x{k} offsets")
            m_tiles = m // 16
            n_tiles = 0
        elif role == 1:
            k, n = info.shape
            if n < 16 or n % 16 != 0:
                raise NotImplementedError("wave_amd tt.dot B operand N dimension must be a positive multiple of 16")
            fixed_affine = _AffineIndex(0, (1, k))
            if ptr.pointer.ptr_offset_affine != fixed_affine and dot_layout is None:
                self._expect_dot_pointer_layout(ptr, info.shape, fixed_affine,
                                                f"B operand K-contiguous column-major {k}x{n} offsets")
            m_tiles = 0
            n_tiles = n // 16
        else:
            raise AssertionError(f"unexpected dot operand role {role}")
        if k < 16 or k % 16 != 0:
            raise NotImplementedError("wave_amd tt.dot operand K dimension must be a positive multiple of 16")
        if dot_layout is not None and dot_layout.shape != info.shape:
            raise NotImplementedError("wave_amd tt.dot symbolic pointer layout must match the operand shape")

        lane_mod = self._lane_mod_index()
        lane_sym = self._sym("lane_mod")
        index_type = self.dsl.simd_type(self.dsl.index_type(), self.width)
        fragments = []
        tokens = []
        k_steps = k // 16
        outer_tiles = m_tiles if role == 0 else n_tiles
        for tile in range(outer_tiles):
            tile_base = tile * 16 * k
            for step in range(k_steps):
                if dot_layout is not None and ptr.pointer.ptr_offset_affine != fixed_affine:
                    if role == 0:
                        coords = (tile * 16 + lane_sym, step * 16)
                    else:
                        coords = (step * 16, tile * 16 + lane_sym)
                    index_expr = dot_layout.expr(coords)
                    bindings = {lane_sym: lane_mod, **dot_layout.offset.bindings}
                else:
                    offset = tile_base + step * 16
                    index_expr = lane_sym * k + offset
                    bindings = {lane_sym: lane_mod}
                index = self.func.index_expr(index_expr, bindings, index_type)
                frag_ptr = self.func.ptr_add(ptr.pointer.ptr_base, index)
                fragment, token = self._emit_masked_dot_fragment_load(frag_ptr, role, info.elem_type, mask, tile, step)
                fragments.append(fragment)
                tokens.append(token)
        return _DotFragmentGrid(
            role=role,
            m_tiles=m_tiles,
            n_tiles=n_tiles,
            k_steps=k_steps,
            fragments=tuple(fragments),
            tokens=tuple(tokens),
        )

    def _emit_masked_dot_fragment_load(self, frag_ptr, role: int, elem_type: str, mask: Optional[_Value] = None,
                                       tile: int = 0, step: int = 0):
        frag_type = self._dot_fragment_type(role=role, elem_type=elem_type)
        if mask is None:
            return self.func.fragment_load(frag_ptr, frag_type)
        frag = self.dsl.FragmentType(frag_type)
        tuple_type = self.dsl.simd_type(self.dsl.vector_type(frag.registers, self.dsl.i32()), width=frag.wave_size)
        regs, token = self.func.load(frag_ptr, tuple_type)
        lane_mod = self._lane_mod_index()
        lane_sym = self._sym("lane_mod")
        coord_bindings = {lane_sym: lane_mod}
        value_type = self.dsl.simd_type(self.dsl.i32(), self.width)
        zero = self._splat_i32(0)
        low_half = self._splat_i32(0x0000FFFF)
        high_half = self._splat_i32(-0x00010000)
        masked_regs = []
        for register in range(frag.registers):
            if role == 0:
                low_coords = (tile * 16 + lane_sym, step * 16 + register * 2)
                high_coords = (tile * 16 + lane_sym, step * 16 + register * 2 + 1)
            else:
                low_coords = (step * 16 + register * 2, tile * 16 + lane_sym)
                high_coords = (step * 16 + register * 2 + 1, tile * 16 + lane_sym)
            value = self.dsl.wave.ExtractOp(value_type, regs, register).result
            low_conditions = self._mask_conditions_for_coords(mask, low_coords, coord_bindings, register)
            high_conditions = self._mask_conditions_for_coords(mask, high_coords, coord_bindings, register)
            if low_conditions or high_conditions:
                valid_low = self._combine_mask_conditions(low_conditions) if low_conditions else None
                valid_high = self._combine_mask_conditions(high_conditions) if high_conditions else None
                keep_low = self.func.select(valid_low, low_half, zero) if valid_low is not None else low_half
                keep_high = self.func.select(valid_high, high_half, zero) if valid_high is not None else high_half
                keep = self.func.binary("ori", keep_low, keep_high)
                value = self.func.binary("andi", value, keep)
            masked_regs.append(value)
        packed = self.dsl.wave.PackOp(tuple_type, masked_regs).result
        return self.func.fragment_pack(packed, frag_type), token

    def _mask_conditions_for_coords(self, mask: _Value, coords: Tuple[object, ...],
                                    coord_bindings: Dict[object, object], register: int) -> Tuple[object, ...]:
        if mask.mask.mask_components:
            conditions = []
            for component in mask.mask.mask_components:
                conditions.extend(self._mask_conditions_for_coords(component, coords, coord_bindings, register))
            return tuple(conditions)
        if mask.mask.cmpi_predicate is not None and mask.mask.cmpi_lhs is not None and mask.mask.cmpi_rhs is not None:
            lhs = self._materialize_symbolic_value_at_coords(mask.mask.cmpi_lhs, coords, coord_bindings)
            rhs = self._materialize_symbolic_value_at_coords(mask.mask.cmpi_rhs, coords, coord_bindings)
            if lhs is not None and rhs is not None:
                return (self.func.cmpi(mask.mask.cmpi_predicate, lhs, rhs), )
        return self._mask_conditions(mask, register)

    def _materialize_symbolic_value_at_coords(self, value: _Value, coords: Tuple[object, ...],
                                              coord_bindings: Dict[object, object]):
        sym_index = value.symbolic.sym_index
        if sym_index is not None:
            if len(sym_index.coeffs) == 0:
                expr = sym_index.const
            elif len(sym_index.coeffs) == len(coords):
                expr = sym_index.const
                for coord, coeff in zip(coords, sym_index.coeffs):
                    expr = expr + coord * coeff
            else:
                return None
            return self._materialize_mask_index_expr(expr, {**coord_bindings, **sym_index.bindings})
        expr, bindings = self._expr_and_bindings(value)
        if expr is not None:
            return self._materialize_mask_index_expr(expr, {**coord_bindings, **bindings})
        return None

    def _materialize_mask_index_expr(self, expr, bindings: Dict[object, object]):
        value = self.func.index_expr(expr, bindings)
        if str(value.type) == "index":
            return self.func.splat(value, self.dsl.index_type(), self.width)
        return value

    def _combine_mask_conditions(self, conditions: Tuple[object, ...]):
        combined = conditions[0]
        for condition in conditions[1:]:
            combined = self.func.select(combined, condition, combined)
        return combined

    def _mask_conditions(self, mask: _Value, register: int) -> Tuple[object, ...]:
        if mask.mask.mask_components:
            conditions = []
            for component in mask.mask.mask_components:
                conditions.extend(self._mask_conditions(component, register))
            return tuple(conditions)
        if mask.lanes.waves:
            return (mask.lanes.waves[min(register, len(mask.lanes.waves) - 1)], )
        if mask.lanes.wave is not None:
            return (mask.lanes.wave, )
        return ()

    def _emit_masked_results(self, conditions: Tuple[object, ...], result_types, emit_then, emit_else):
        condition = conditions[0]
        if len(conditions) == 1:
            with self.func.where(condition, result_types) as where_op:
                self.func.yield_(emit_then())
        else:
            with self.func.where(condition, result_types) as where_op:
                self.func.yield_(self._emit_masked_results(conditions[1:], result_types, emit_then, emit_else))
        block = where_op.elseRegion.blocks.append()
        with self.dsl.InsertionPoint(block):
            self.dsl.wave.YieldOp(list(emit_else()))
        return where_op.results

    def _emit_under_mask(self, conditions: Tuple[object, ...], emit_then) -> None:
        if not conditions:
            emit_then()
            return
        with self.func.where(conditions[0]):
            self._emit_under_mask(conditions[1:], emit_then)

    def _expect_zero_dot_load_other(self, other: _Value, elem_type: str) -> None:
        if other.constant.splat_const is None:
            raise NotImplementedError("wave_amd masked tt.dot operand loads require zero-splat `other`")
        value, other_elem_type, _ = other.constant.splat_const
        if value != 0 or other_elem_type not in {elem_type, "f32"}:
            raise NotImplementedError("wave_amd masked tt.dot operand loads require zero-splat `other`")

    def _emit_fragment_store(self, ptr: _Value, value: _Value, mask: Optional[_Value] = None) -> None:
        if ptr.pointer.ptr_base is None:
            raise NotImplementedError("wave_amd tt.dot result stores require a splatted pointer base")
        if value.shape is None or len(value.shape) != 2:
            raise NotImplementedError("wave_amd tt.dot result stores require a rank-2 matrix result")
        m, n = value.shape
        fixed_affine = _AffineIndex(0, (n, 1))
        dot_layout = ptr.pointer.ptr_dot_layout
        if ptr.pointer.ptr_offset_affine != fixed_affine and dot_layout is None:
            self._expect_dot_pointer_layout(ptr, value.shape, fixed_affine, f"C result row-major {m}x{n} offsets")
        if mask is not None and (value.dot.dot_grid is None or value.dot.dot_grid.m_tiles != 1
                                 or value.dot.dot_grid.n_tiles != 1):
            raise NotImplementedError("wave_amd masked tt.dot fragment stores currently support one 16x16 tile")
        if value.dot.dot_grid is not None:
            grid = value.dot.dot_grid
            if grid.role != 2:
                raise NotImplementedError("wave_amd tt.dot result stores require C/result fragments")
            for m_tile in range(grid.m_tiles):
                for n_tile in range(grid.n_tiles):
                    tile_base = m_tile * 16 * n + n_tile * 16
                    owner_conditions = self._dot_tile_owner_conditions(grid, m_tile, n_tile)
                    self._emit_dot_row_major_fragment_store(
                        ptr.pointer.ptr_base, grid.c(m_tile, n_tile), n, tile_base,
                        dot_layout if ptr.pointer.ptr_offset_affine != fixed_affine else None, m_tile, n_tile, mask,
                        owner_conditions)
        else:
            self._emit_dot_row_major_fragment_store(
                ptr.pointer.ptr_base, value.dot.fragment, n, 0,
                dot_layout if ptr.pointer.ptr_offset_affine != fixed_affine else None, 0, 0, mask,
                self._dot_tile_owner_conditions(None, 0, 0))
        self.load_tokens.clear()

    def _emit_dot_row_major_fragment_store(
        self, ptr_base, fragment, leading_dim: int, tile_base: int, dot_layout: Optional[_DotPointerLayout] = None,
        m_tile: int = 0, n_tile: int = 0, mask: Optional[_Value] = None, owner_conditions: Tuple[object,
                                                                                                 ...] = ()) -> None:
        regs = self.func.fragment_unpack(fragment)
        lane_mod = self._lane_mod_index()
        lane_sym = self._sym("lane_mod")
        row_parity = self._row_parity_index()
        row_parity_sym = self._sym("row_parity")
        after = self._load_after_token()
        value_type = self.dsl.simd_type(self.dsl.i32(), self.width)
        index_type = self.dsl.simd_type(self.dsl.index_type(), self.width)
        for register in range(8):
            if dot_layout is not None:
                row = m_tile * 16 + register * 2 + row_parity_sym
                col = n_tile * 16 + lane_sym
                row_major = dot_layout.expr((row, col))
                bindings = {lane_sym: lane_mod, row_parity_sym: row_parity, **dot_layout.offset.bindings}
            else:
                row_major = tile_base + (register * 2 + row_parity_sym) * leading_dim + lane_sym
                bindings = {lane_sym: lane_mod, row_parity_sym: row_parity}
            index = self.func.index_expr(row_major, bindings, index_type)
            value = self.dsl.wave.ExtractOp(value_type, regs, register).result
            ptr = self.func.ptr_add(ptr_base, index)
            if mask is None:
                if owner_conditions:
                    self._emit_under_mask(owner_conditions,
                                          lambda value=value, ptr=ptr: self.func.store(value, ptr, after=after))
                else:
                    self.func.store(value, ptr, after=after)
            else:
                conditions = self._mask_conditions(mask, register)
                if not conditions:
                    raise NotImplementedError("wave_amd masked tt.dot fragment stores require a Wave mask")
                self._emit_under_mask(owner_conditions + conditions,
                                      lambda value=value, ptr=ptr: self.func.store(value, ptr, after=after))

    def _dot_tile_owner_conditions(self, grid: Optional[_DotFragmentGrid], m_tile: int,
                                   n_tile: int) -> Tuple[object, ...]:
        if self.num_warps == 1 and self.num_ctas == 1:
            return ()
        n_tiles = 1 if grid is None else grid.n_tiles
        tile_index = m_tile * n_tiles + n_tile
        workers = self.num_warps * self.num_ctas
        owner = tile_index % workers
        conditions = []
        if self.num_warps > 1:
            conditions.append(self._wave_owner_condition(owner % self.num_warps))
        if self.num_ctas > 1:
            conditions.append(self._cta_owner_condition(owner // self.num_warps))
        return tuple(conditions)

    def _wave_owner_condition(self, owner: int):
        workitem = self._workitem_id()
        wave_id = self.func.binary("shri", workitem, self._splat_i32(_log2_int(self.width)))
        return self.func.cmpi("eq", wave_id, self._splat_i32(owner))

    def _cta_owner_condition(self, owner: int):
        cta_wave = self.func.splat(self._workgroup_id(0), self.dsl.i32(), self.width)
        cta_wave = self.func.binary("andi", cta_wave, self._splat_i32(self.num_ctas - 1))
        return self.func.cmpi("eq", cta_wave, self._splat_i32(owner))

    def _expect_dot_pointer_layout(self, ptr: _Value, shape: Optional[Tuple[int, ...]], expected: _AffineIndex,
                                   description: str) -> None:
        if shape is None:
            raise NotImplementedError(f"wave_amd tt.dot currently supports {description}")
        if ptr.pointer.ptr_offset_affine != expected:
            raise NotImplementedError(f"wave_amd tt.dot currently supports only {description}")

    def _dot_fragment_type(self, role: int, elem_type: str):
        registers = 8
        return self.dsl.fragment_type(
            role,
            self._scalar_type(elem_type),
            rows=16,
            columns=16,
            wave_size=self.width,
            registers=registers,
        )

    def _load_after_token(self):
        if len(self.load_tokens) == 1:
            return self.load_tokens[0]
        if len(self.load_tokens) > 1:
            return self.func.join(*self.load_tokens)
        return None

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
        self.func.store(value_wave, ptr_wave, after=self._load_after_token())

    def _emit_unmasked_store_of_masked_load_other(self, ptr: _Value, value: _Value) -> None:
        other_waves = value.mask.other_waves or (value.mask.other_wave, )
        if value.mask.other_splat_const is not None and len(other_waves) != len(self._waves(ptr)):
            other_waves = tuple(
                self._materialize_splat_constant(value.mask.other_splat_const) for _ in range(len(self._waves(ptr))))
        for mask_wave, ptr_wave, value_wave, other_wave, token in zip(value.mask.mask_waves or (value.mask.mask_wave, ),
                                                                      self._waves(ptr), self._waves(value), other_waves,
                                                                      value.memory.tokens or (value.memory.token, )):
            with self.func.where(mask_wave, [self.dsl.mem_token_type()]) as where_op:
                token = self.func.store(value_wave, ptr_wave, after=token)
                self.func.yield_([token])
            block = where_op.elseRegion.blocks.append()
            with self.dsl.InsertionPoint(block):
                token = self.func.store(other_wave, ptr_wave)
                self.dsl.wave.YieldOp([token])
        self.load_tokens.clear()

    def _load_other_value(self, other: _Value, elem_type: str) -> _Value:
        if other.lanes.wave is None:
            raise NotImplementedError("wave_amd masked tt.load `other` requires a Wave value")
        if other.lanes.waves:
            return other
        splat_const = None
        if other.constant.splat_const is not None and other.constant.splat_const[2] is None:
            splat_const = (other.constant.splat_const[0], other.constant.splat_const[1], None)
        return _Value(
            elem_type=other.elem_type or elem_type,
            lanes=_LanePayload(wave=other.lanes.wave),
            constant=_ConstantPayload(splat_const=splat_const),
        )

    def _materialize_splat_constant(self, constant: Tuple[object, str, Optional[int]]):
        value, elem_type, width = constant
        scalar = self.func.constant(self._scalar_type(elem_type), value)
        if width is None:
            return scalar
        return self.func.splat(scalar, self._scalar_type(elem_type), width)

    def _negative_one_like(self, value: _Value, index: int = 0):
        elem_type = value.elem_type or "i32"
        scalar = self.func.constant(self._scalar_type(elem_type), -1)
        wave = value.lanes.waves[index] if value.lanes.waves else value.lanes.wave
        if str(wave.type).startswith("!wave.simd<"):
            return self.func.splat(scalar, self._scalar_type(elem_type), self.width)
        return scalar

    def _select_operands(self, true_value: _Value, false_value: _Value):
        true_wave = true_value.lanes.wave
        false_wave = false_value.lanes.wave
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
        if state.indexing.index is not None and index == 0:
            return state.indexing.index
        if state.indexing.indexes:
            return state.indexing.indexes[index]
        expr, bindings = self._expr_and_bindings(state, index)
        if expr is None:
            raise NotImplementedError("wave_amd requires symbolic tt.addptr offsets")
        return self.func.index_expr(expr, bindings, self.dsl.simd_type(self.dsl.index_type(), self.width))

    def _indexes(self, state: _Value, layout: _BlockedLayout):
        if state.indexing.indexes:
            return state.indexing.indexes
        indexes = tuple(self._index(state, index) for index in range(layout.registers))
        state.indexing.indexes = indexes
        state.indexing.index = indexes[0]
        return indexes

    def _expr_and_bindings(self, state: _Value, index: int = 0):
        if state.symbolic.exprs:
            return state.symbolic.exprs[index], dict(state.symbolic.bindings_by_wave[index])
        if state.symbolic.expr is not None:
            return state.symbolic.expr, dict(state.symbolic.bindings)
        if state.constant.const is None:
            return None, {}

        name = f"c{state.constant.const}"
        sym = self._sym(name)
        return sym, {sym: state.lanes.wave}

    def _lane_id(self):
        if self._lane_value is None:
            self._lane_value = self.func.lane_id(self.dsl.i32(), self.width)
        return self._lane_value

    def _lane_mod_index(self):
        return self.func.binary("andi", self._lane_id(), self._splat_i32(15))

    def _row_parity_index(self):
        return self.func.binary("shri", self._lane_id(), self._splat_i32(4))

    def _thread_id_and_sym(self):
        if self.num_warps == 1:
            return self._lane_id(), self._sym("lid")
        return self._workitem_id(), self._sym("wi")

    def _workitem_id(self):
        if self._workitem_value is None:
            self._workitem_value = self.func.workitem_id(axis=0, element_type=self.dsl.i32(), width=self.width)
        return self._workitem_value

    def _workgroup_id(self, axis: int):
        if axis not in self._workgroup_values:
            self._workgroup_values[axis] = self.func.workgroup_id(axis)
        return self._workgroup_values[axis]

    def _workgroup_sym(self, axis: int):
        return self._sym(f"wg_{axis}")

    def _cta_expr_and_bindings(self):
        if self.num_ctas == 1:
            return None, {}
        wg_sym = self._workgroup_sym(0)
        return self.dsl.mod(wg_sym, self.num_ctas), {wg_sym: self._workgroup_id(0)}

    def _splat_i32(self, value: int):
        scalar = self.func.constant(self.dsl.i32(), value)
        return self.func.splat(scalar, self.dsl.i32(), self.width)

    def _waves(self, state: _Value) -> Tuple[object, ...]:
        if state.lanes.waves:
            return state.lanes.waves
        if state.lanes.wave is None:
            return ()
        return (state.lanes.wave, )

    def _can_emit_binary_waves(self, lhs: _Value, rhs: _Value) -> bool:
        return bool(lhs.lanes.waves and rhs.lanes.waves and len(lhs.lanes.waves) == len(rhs.lanes.waves)
                    and all(lhs_wave.type == rhs_wave.type
                            for lhs_wave, rhs_wave in zip(lhs.lanes.waves, rhs.lanes.waves)))

    def _coerce_tensor(self, state: _Value, layout: _BlockedLayout, info: _TensorInfo) -> _Value:
        if state.layout is not None:
            if state.layout != layout:
                raise NotImplementedError("wave_amd elementwise operands must currently share one layout")
            return state
        if state.lanes.waves and len(state.lanes.waves) == layout.registers:
            state.layout = layout
            state.shape = info.shape
            return state
        if state.lanes.wave is None:
            raise NotImplementedError("wave_amd cannot broadcast an unsupported tensor value")
        waves = tuple(
            self.func.splat(state.lanes.wave, self._scalar_type(state.elem_type or info.elem_type), self.width)
            for _ in range(layout.registers))
        expr, bindings = self._expr_and_bindings(state)
        exprs = tuple(expr for _ in range(layout.registers))
        bindings_by_wave = tuple(dict(bindings) for _ in range(layout.registers))
        return _Value(
            elem_type=state.elem_type or info.elem_type,
            layout=layout,
            shape=info.shape,
            lanes=_LanePayload(wave=waves[0], waves=waves),
            constant=_ConstantPayload(const=state.constant.const, splat_const=state.constant.splat_const),
            symbolic=_SymbolicPayload(affine=_affine_splat(state.symbolic.affine, len(info.shape)), expr=expr,
                                      exprs=exprs, bindings=bindings, bindings_by_wave=bindings_by_wave,
                                      sym_index=_symbolic_splat(state.symbolic.sym_index, len(info.shape))),
            pointer=_PointerPayload(ptr_base=state.pointer.ptr_base),
            mask=_MaskPayload(mask_id=state.mask.mask_id),
        )

    def _coordinate_value(self, layout: _BlockedLayout, info: _TensorInfo, axis: int, start: int = 0) -> _Value:
        return self._symbolic_tensor_value(layout, info, _symbolic_coord(len(info.shape), axis, start), axis, start,
                                           materialize_waves=len(info.shape) == 1)

    def _symbolic_tensor_value(self, layout: _BlockedLayout, info: _TensorInfo, sym_index: Optional[_SymbolicIndex],
                               coord_axis: Optional[int] = None, coord_start: int = 0,
                               materialize_waves: bool = True) -> _Value:
        if sym_index is None:
            raise NotImplementedError("wave_amd requires symbolic tensor metadata")
        thread, sym = self._thread_id_and_sym()
        cta_expr, cta_bindings = self._cta_expr_and_bindings()
        waves = []
        exprs = []
        bindings_by_wave = []
        for register in range(layout.registers):
            coords = tuple(
                layout.coord_expr(self.dsl, sym, cta_expr, register, axis) for axis in range(len(info.shape)))
            expr = sym_index.const
            for coord, coeff in zip(coords, sym_index.coeffs):
                expr = expr + coord * coeff
            bindings = {sym: thread, **cta_bindings}
            bindings.update(sym_index.bindings)
            if materialize_waves:
                if info.elem_type == "index" and _symbolic_is_uniform(sym_index):
                    scalar = self.func.index_expr(sym_index.const, sym_index.bindings, self.dsl.index_type())
                    wave = self.func.splat(scalar, self.dsl.index_type(), self.width)
                elif coord_axis is not None and len(layout.shape) == 1 and not sym_index.bindings:
                    offset = register * layout.threads_per_cta + coord_start
                    if cta_expr is None:
                        wave = thread if offset == 0 else self.func.addi(thread, self._splat_i32(offset))
                    else:
                        wave = self.func.index_expr(expr, bindings,
                                                    self.dsl.simd_type(self.dsl.index_type(), self.width))
                else:
                    index_wave = self.func.index_expr(expr, bindings,
                                                      self.dsl.simd_type(self.dsl.index_type(), self.width))
                    if info.elem_type == "index":
                        wave = index_wave
                    else:
                        wave = self.func.cast(index_wave,
                                              self.dsl.simd_type(self._scalar_type(info.elem_type), self.width),
                                              self.dsl.CastKind.IntConvert)
                waves.append(wave)
            exprs.append(expr)
            bindings_by_wave.append(bindings)
        return _Value(
            elem_type=info.elem_type,
            layout=layout,
            shape=info.shape,
            lanes=_LanePayload(wave=waves[0] if waves else None, waves=tuple(waves)),
            symbolic=_SymbolicPayload(
                affine=_affine_coord(len(info.shape), coord_axis, coord_start) if coord_axis is not None else None,
                expr=exprs[0], exprs=tuple(exprs), bindings=bindings_by_wave[0],
                bindings_by_wave=tuple(bindings_by_wave), sym_index=sym_index, coord_axis=coord_axis,
                coord_start=coord_start),
        )

    def _result_tensor_info(self, op) -> Optional[_TensorInfo]:
        info = _wave_amd_native().get_result_tensor_info(op, 0)
        return _pack_tensor_info(info)

    def _value_tensor_info(self, value) -> Optional[_TensorInfo]:
        shape = tuple(int(dim) for dim in value.get_shape())
        if not shape:
            return None
        elem_type = _tensor_element_type_name(value.get_type())
        if elem_type is None:
            return None
        return _TensorInfo(shape, elem_type)

    def _layout_for_info(self, info: Optional[_TensorInfo]) -> Optional[_BlockedLayout]:
        if info is None:
            return None
        return _BlockedLayout.for_tensor(info, self.width, self.num_warps, self.num_ctas)

    def _can_defer_coordinate_layout(self, info: _TensorInfo) -> bool:
        elements = _product(info.shape)
        return (elements < self.width * self.num_warps * self.num_ctas
                and all(_is_power_of_two(dim) for dim in info.shape) and _is_power_of_two(elements))

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
    mask_ids = {state.mask.mask_id for state in states if state.mask.mask_id is not None}
    if not mask_ids:
        return None
    if len(mask_ids) != 1:
        raise NotImplementedError("wave_amd masked values must share one SSA mask")
    return next(iter(mask_ids))


def _affine_coord(rank: int, axis: int, start: int = 0) -> _AffineIndex:
    coeffs = [0] * rank
    coeffs[axis] = 1
    return _AffineIndex(start, tuple(coeffs))


def _affine_splat(value: Optional[_AffineIndex], rank: int) -> Optional[_AffineIndex]:
    if value is None:
        return None
    if value.coeffs and len(value.coeffs) != rank:
        return None
    return _AffineIndex(value.const, (0, ) * rank)


def _affine_add(lhs: Optional[_AffineIndex], rhs: Optional[_AffineIndex]) -> Optional[_AffineIndex]:
    lhs, rhs = _affine_common_rank(lhs, rhs)
    if lhs is None or rhs is None:
        return None
    return _AffineIndex(lhs.const + rhs.const, tuple(l + r for l, r in zip(lhs.coeffs, rhs.coeffs)))


def _affine_sub(lhs: Optional[_AffineIndex], rhs: Optional[_AffineIndex]) -> Optional[_AffineIndex]:
    lhs, rhs = _affine_common_rank(lhs, rhs)
    if lhs is None or rhs is None:
        return None
    return _AffineIndex(lhs.const - rhs.const, tuple(l - r for l, r in zip(lhs.coeffs, rhs.coeffs)))


def _affine_mul(lhs: Optional[_AffineIndex], rhs: Optional[_AffineIndex]) -> Optional[_AffineIndex]:
    lhs, rhs = _affine_common_rank(lhs, rhs)
    if lhs is None or rhs is None:
        return None
    if _affine_is_constant(lhs):
        return _affine_scale(rhs, lhs.const)
    if _affine_is_constant(rhs):
        return _affine_scale(lhs, rhs.const)
    return None


def _affine_common_rank(lhs: Optional[_AffineIndex],
                        rhs: Optional[_AffineIndex]) -> Tuple[Optional[_AffineIndex], Optional[_AffineIndex]]:
    if lhs is None or rhs is None:
        return lhs, rhs
    lhs_rank = len(lhs.coeffs)
    rhs_rank = len(rhs.coeffs)
    if lhs_rank == rhs_rank:
        return lhs, rhs
    if lhs_rank == 0:
        return _AffineIndex(lhs.const, (0, ) * rhs_rank), rhs
    if rhs_rank == 0:
        return lhs, _AffineIndex(rhs.const, (0, ) * lhs_rank)
    return None, None


def _affine_is_constant(value: _AffineIndex) -> bool:
    return all(coeff == 0 for coeff in value.coeffs)


def _affine_scale(value: _AffineIndex, scale: int) -> _AffineIndex:
    return _AffineIndex(value.const * scale, tuple(coeff * scale for coeff in value.coeffs))


def _symbolic_coord(rank: int, axis: int, start: int = 0) -> _SymbolicIndex:
    coeffs = [0] * rank
    coeffs[axis] = 1
    return _SymbolicIndex(start, tuple(coeffs), {})


def _symbolic_splat(value: Optional[_SymbolicIndex], rank: int) -> Optional[_SymbolicIndex]:
    if value is None:
        return None
    if len(value.coeffs) == rank:
        return value
    if value.coeffs and not _symbolic_is_uniform(value):
        return None
    return _SymbolicIndex(value.const, (0, ) * rank, dict(value.bindings))


def _symbolic_expand_dims(value: Optional[_SymbolicIndex], axis: int, rank: int) -> Optional[_SymbolicIndex]:
    if value is None or len(value.coeffs) != rank - 1:
        return None
    coeffs = list(value.coeffs)
    coeffs.insert(axis, 0)
    return _SymbolicIndex(value.const, tuple(coeffs), dict(value.bindings))


def _symbolic_broadcast(value: Optional[_SymbolicIndex], rank: int) -> Optional[_SymbolicIndex]:
    if value is None:
        return None
    if len(value.coeffs) == rank:
        return value
    if _symbolic_is_uniform(value):
        return _SymbolicIndex(value.const, (0, ) * rank, dict(value.bindings))
    return None


def _symbolic_add(lhs: Optional[_SymbolicIndex], rhs: Optional[_SymbolicIndex]) -> Optional[_SymbolicIndex]:
    lhs, rhs = _symbolic_common_rank(lhs, rhs)
    if lhs is None or rhs is None:
        return None
    return _SymbolicIndex(lhs.const + rhs.const, tuple(l + r for l, r in zip(lhs.coeffs, rhs.coeffs)),
                          {**lhs.bindings, **rhs.bindings})


def _symbolic_sub(lhs: Optional[_SymbolicIndex], rhs: Optional[_SymbolicIndex]) -> Optional[_SymbolicIndex]:
    lhs, rhs = _symbolic_common_rank(lhs, rhs)
    if lhs is None or rhs is None:
        return None
    return _SymbolicIndex(lhs.const - rhs.const, tuple(l - r for l, r in zip(lhs.coeffs, rhs.coeffs)),
                          {**lhs.bindings, **rhs.bindings})


def _symbolic_mul(lhs: Optional[_SymbolicIndex], rhs: Optional[_SymbolicIndex]) -> Optional[_SymbolicIndex]:
    lhs, rhs = _symbolic_common_rank(lhs, rhs)
    if lhs is None or rhs is None:
        return None
    if _symbolic_is_uniform(lhs):
        return _symbolic_scale(rhs, lhs.const, lhs.bindings)
    if _symbolic_is_uniform(rhs):
        return _symbolic_scale(lhs, rhs.const, rhs.bindings)
    return None


def _symbolic_common_rank(lhs: Optional[_SymbolicIndex],
                          rhs: Optional[_SymbolicIndex]) -> Tuple[Optional[_SymbolicIndex], Optional[_SymbolicIndex]]:
    if lhs is None or rhs is None:
        return lhs, rhs
    lhs_rank = len(lhs.coeffs)
    rhs_rank = len(rhs.coeffs)
    if lhs_rank == rhs_rank:
        return lhs, rhs
    if lhs_rank == 0:
        return _SymbolicIndex(lhs.const, (0, ) * rhs_rank, dict(lhs.bindings)), rhs
    if rhs_rank == 0:
        return lhs, _SymbolicIndex(rhs.const, (0, ) * lhs_rank, dict(rhs.bindings))
    if _symbolic_is_uniform(lhs):
        return _SymbolicIndex(lhs.const, (0, ) * rhs_rank, dict(lhs.bindings)), rhs
    if _symbolic_is_uniform(rhs):
        return lhs, _SymbolicIndex(rhs.const, (0, ) * lhs_rank, dict(rhs.bindings))
    return None, None


def _symbolic_is_uniform(value: _SymbolicIndex) -> bool:
    return all(coeff == 0 for coeff in value.coeffs)


def _symbolic_scale(value: _SymbolicIndex, scale, scale_bindings: Dict[object, object]) -> _SymbolicIndex:
    return _SymbolicIndex(value.const * scale, tuple(coeff * scale for coeff in value.coeffs),
                          {**value.bindings, **scale_bindings})


def _pack_tensor_info(info) -> Optional[_TensorInfo]:
    if info is None:
        return None
    shape, elem_type, is_pointer = info
    return _TensorInfo(tuple(int(dim) for dim in shape), str(elem_type), bool(is_pointer))


def _tensor_element_type_name(tensor_type) -> Optional[str]:
    text = str(tensor_type)
    if not text.startswith("tensor<") or not text.endswith(">"):
        return None
    body = text[len("tensor<"):-1]
    element = body.rsplit("x", 1)[-1].split(",", 1)[0].strip()
    if element in {"f16", "bf16", "f32", "i1", "i8", "i32", "i64"}:
        return element
    return None


def _value_element_type_name(value_type) -> Optional[str]:
    text = str(value_type)
    if text in {"index", "i1", "i8", "i32", "i64", "f16", "bf16", "f32"}:
        return text
    return _tensor_element_type_name(value_type)


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

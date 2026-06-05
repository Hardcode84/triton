from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class _TensorInfo:
    shape: Tuple[int, ...]
    elem_type: str
    is_pointer: bool = False


@dataclass(frozen=True)
class _Affine:
    const: int
    coeffs: Tuple[int, ...]

    def to_dict(self):
        return {"const": self.const, "coeffs": list(self.coeffs)}


@dataclass(frozen=True)
class _State:
    info: Optional[_TensorInfo] = None
    const: Optional[int] = None
    affine: Optional[_Affine] = None
    pointer_base_id: Optional[int] = None


def analyze_wave_gemm(module) -> Dict[str, object]:
    """Analyze Triton GEMM structure without emitting Wave IR.

    This intentionally records metadata only.  The current Wave lowering may
    continue to rediscover these facts until the translator is made dumb.
    """

    entry_name = module.get_entry_func_name()
    if not entry_name:
        return {"entry": None, "dots": []}
    ops = _collect_entry_ops(module, entry_name)
    states: Dict[int, _State] = {}
    producers = {}
    dots = []
    dot_by_result = {}

    for op in ops:
        for result_index in range(op.get_num_results()):
            producers[op.get_result(result_index).id()] = op

        name = op.get_name()
        if name == "arith.constant":
            _set_result(states, op, _constant_state(op))
        elif name == "tt.make_range":
            _set_result(states, op, _make_range_state(op))
        elif name == "tt.expand_dims":
            _set_result(states, op, _expand_dims_state(op, states))
        elif name == "tt.broadcast":
            _set_result(states, op, _broadcast_state(op, states))
        elif name == "tt.splat":
            _set_result(states, op, _splat_state(op, states))
        elif name == "arith.addi":
            _set_result(states, op, _binary_state(op, states, _affine_add))
        elif name == "arith.muli":
            _set_result(states, op, _muli_state(op, states))
        elif name == "tt.addptr":
            _set_result(states, op, _addptr_state(op, states))
        elif name == "tt.load":
            _set_result(states, op, _load_state(op))
        elif name == "ttg.convert_layout":
            _set_result(states, op, states.get(op.get_operand(0).id(), _State()))
        elif name == "tt.dot":
            dot = _dot_metadata(op, states, producers, len(dots))
            dot_by_result[op.get_result(0).id()] = dot
            dots.append(dot)
            _set_result(states, op, _State(info=_result_tensor_info(op)))
        elif name == "tt.store":
            _attach_store_metadata(op, states, producers, dot_by_result)

    return {"entry": entry_name, "dots": dots}


def _collect_entry_ops(module, entry_name: str) -> Sequence[object]:
    pending_body_ops = []
    entry_ops = []

    def visit(op):
        nonlocal entry_ops
        name = op.get_name()
        if name == "builtin.module":
            return
        if name == "tt.func":
            if op.get_str_attr("sym_name") == entry_name:
                entry_ops = list(pending_body_ops)
            pending_body_ops.clear()
            return
        pending_body_ops.append(op)

    module.walk(visit)
    return entry_ops


def _set_result(states: Dict[int, _State], op, state: _State) -> None:
    if op.get_num_results() > 0:
        states[op.get_result(0).id()] = state


def _constant_state(op) -> _State:
    value = op.get_constant_value()
    return _State(info=_result_tensor_info(op), const=value if isinstance(value, int) else None)


def _make_range_state(op) -> _State:
    start = op.get_int_attr("start")
    return _State(info=_result_tensor_info(op), affine=_Affine(start, (1, )))


def _expand_dims_state(op, states: Dict[int, _State]) -> _State:
    src = states.get(op.get_operand(0).id(), _State())
    info = _result_tensor_info(op)
    affine = None
    if src.affine is not None and info is not None:
        axis = op.get_int_attr("axis")
        coeffs = list(src.affine.coeffs)
        coeffs.insert(axis, 0)
        affine = _Affine(src.affine.const, tuple(coeffs))
    return _State(info=info, const=src.const, affine=affine, pointer_base_id=src.pointer_base_id)


def _broadcast_state(op, states: Dict[int, _State]) -> _State:
    src = states.get(op.get_operand(0).id(), _State())
    info = _result_tensor_info(op)
    affine = src.affine
    if affine is not None and info is not None and len(affine.coeffs) != len(info.shape):
        affine = None
    return _State(info=info, const=src.const, affine=affine, pointer_base_id=src.pointer_base_id)


def _splat_state(op, states: Dict[int, _State]) -> _State:
    src = states.get(op.get_operand(0).id(), _State())
    return _State(info=_result_tensor_info(op), const=src.const, pointer_base_id=src.pointer_base_id)


def _binary_state(op, states: Dict[int, _State], affine_builder) -> _State:
    lhs = states.get(op.get_operand(0).id(), _State())
    rhs = states.get(op.get_operand(1).id(), _State())
    affine = affine_builder(lhs.affine, rhs.affine)
    return _State(info=_result_tensor_info(op), affine=affine)


def _muli_state(op, states: Dict[int, _State]) -> _State:
    lhs = states.get(op.get_operand(0).id(), _State())
    rhs = states.get(op.get_operand(1).id(), _State())
    affine = None
    if lhs.affine is not None and rhs.const is not None:
        affine = _affine_scale(lhs.affine, rhs.const)
    elif rhs.affine is not None and lhs.const is not None:
        affine = _affine_scale(rhs.affine, lhs.const)
    return _State(info=_result_tensor_info(op), affine=affine)


def _addptr_state(op, states: Dict[int, _State]) -> _State:
    base_id = op.get_operand(0).id()
    base = states.get(base_id, _State())
    offset = states.get(op.get_operand(1).id(), _State())
    pointer_base_id = base.pointer_base_id if base.pointer_base_id is not None else base_id
    return _State(info=_result_tensor_info(op), affine=offset.affine, pointer_base_id=pointer_base_id)


def _load_state(op) -> _State:
    return _State(info=_result_tensor_info(op))


def _affine_add(lhs: Optional[_Affine], rhs: Optional[_Affine]) -> Optional[_Affine]:
    if lhs is None or rhs is None or len(lhs.coeffs) != len(rhs.coeffs):
        return None
    return _Affine(lhs.const + rhs.const, tuple(a + b for a, b in zip(lhs.coeffs, rhs.coeffs)))


def _affine_scale(affine: _Affine, scale: int) -> _Affine:
    return _Affine(affine.const * scale, tuple(coeff * scale for coeff in affine.coeffs))


def _dot_metadata(op, states: Dict[int, _State], producers: Dict[int, object], index: int) -> Dict[str, object]:
    lhs_info = _value_tensor_info(op.get_operand(0))
    rhs_info = _value_tensor_info(op.get_operand(1))
    result_info = _result_tensor_info(op)
    return dict(
        index=index,
        op="tt.dot",
        lhs=_tensor_info_dict(lhs_info),
        rhs=_tensor_info_dict(rhs_info),
        result=_tensor_info_dict(result_info),
        operands=[
            _dot_operand_metadata("A", op.get_operand(0), lhs_info, states, producers, result_info),
            _dot_operand_metadata("B", op.get_operand(1), rhs_info, states, producers, result_info),
        ],
        store=None,
    )


def _dot_operand_metadata(role: str, value, info: Optional[_TensorInfo], states: Dict[int, _State],
                          producers: Dict[int, object], result_info: Optional[_TensorInfo]) -> Dict[str, object]:
    source_value_id = _skip_layout_conversions(value.id(), producers)
    producer = producers.get(source_value_id)
    pointer = None
    if producer is not None and producer.get_name() == "tt.load":
        pointer = _pointer_metadata(role, producer.get_operand(0), info, states, producers, result_info)
    return {
        "role": role,
        "value_id": value.id(),
        "source_value_id": source_value_id,
        "producer": None if producer is None else producer.get_name(),
        "tensor": _tensor_info_dict(info),
        "pointer": pointer,
    }


def _attach_store_metadata(op, states: Dict[int, _State], producers: Dict[int, object], dot_by_result) -> None:
    value_id = _skip_layout_conversions(op.get_operand(1).id(), producers)
    dot = dot_by_result.get(value_id)
    if dot is None:
        return
    result_info = _value_tensor_info(op.get_operand(1))
    dot["store"] = _pointer_metadata("C", op.get_operand(0), result_info, states, producers, result_info)


def _pointer_metadata(role: str, ptr_value, info: Optional[_TensorInfo], states: Dict[int, _State],
                      producers: Dict[int, object], result_info: Optional[_TensorInfo]) -> Dict[str, object]:
    source_value_id = _skip_layout_conversions(ptr_value.id(), producers)
    producer = producers.get(source_value_id)
    state = states.get(source_value_id, _State())
    expected = _expected_affine(role, info, result_info)
    actual = state.affine
    classification = "unknown"
    if producer is not None and producer.get_name() == "tt.addptr":
        if expected is not None and actual == expected:
            classification = "dense_static"
        elif actual is None:
            classification = "symbolic_or_dynamic"
        else:
            classification = "affine_mismatch"
    return {
        "role": role,
        "value_id": ptr_value.id(),
        "source_value_id": source_value_id,
        "producer": None if producer is None else producer.get_name(),
        "tensor": _tensor_info_dict(info),
        "base_value_id": state.pointer_base_id,
        "expected_affine": None if expected is None else expected.to_dict(),
        "actual_affine": None if actual is None else actual.to_dict(),
        "classification": classification,
    }


def _expected_affine(role: str, info: Optional[_TensorInfo], result_info: Optional[_TensorInfo]) -> Optional[_Affine]:
    if info is None or len(info.shape) != 2:
        return None
    if role == "A":
        return _Affine(0, (info.shape[1], 1))
    if role == "B":
        return _Affine(0, (1, info.shape[0]))
    if role == "C" and result_info is not None and len(result_info.shape) == 2:
        return _Affine(0, (result_info.shape[1], 1))
    return None


def _skip_layout_conversions(value_id: int, producers: Dict[int, object]) -> int:
    while True:
        producer = producers.get(value_id)
        if producer is None or producer.get_name() != "ttg.convert_layout" or producer.get_num_operands() != 1:
            return value_id
        value_id = producer.get_operand(0).id()


def _result_tensor_info(op) -> Optional[_TensorInfo]:
    if op.get_num_results() == 0:
        return None
    return _tensor_info_from_type(op.get_result(0).get_type())


def _value_tensor_info(value) -> Optional[_TensorInfo]:
    return _tensor_info_from_type(value.get_type())


def _tensor_info_from_type(type_obj) -> Optional[_TensorInfo]:
    text = str(type_obj)
    if not text.startswith("tensor<") or not text.endswith(">"):
        return None
    body = text[len("tensor<"):-1]
    body = body.split(",", 1)[0]
    if "x" not in body:
        return None
    shape_text, elem_type = body.rsplit("x", 1)
    try:
        shape = tuple(int(dim) for dim in shape_text.split("x") if dim)
    except ValueError:
        return None
    if elem_type.startswith("!tt.ptr<") and elem_type.endswith(">"):
        return _TensorInfo(shape, elem_type[len("!tt.ptr<"):-1], True)
    return _TensorInfo(shape, elem_type, False)


def _tensor_info_dict(info: Optional[_TensorInfo]):
    if info is None:
        return None
    return {"shape": list(info.shape), "elem_type": info.elem_type, "is_pointer": info.is_pointer}

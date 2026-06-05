from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Optional, Sequence, Tuple

BUFFER_RANGE_ATTR = "waveamd.buffer.range_bytes"


def _product(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


def _byte_width(elem_type: str) -> int:
    if elem_type in {"f16", "bf16"}:
        return 2
    if elem_type in {"f32", "i32"}:
        return 4
    if elem_type == "i64":
        return 8
    raise NotImplementedError(f"wave_amd buffer planning does not support type {elem_type!r}")


@dataclass(frozen=True)
class WaveGemmPipelinePlan:
    """Backend-side plan for stages reproduced from Triton's AMD GEMM pipeline."""

    use_buffer_ops: bool
    num_stages: int
    num_warps: int
    num_ctas: int


@dataclass(frozen=True)
class WaveBufferPlan:
    elem_type: str
    shape: Tuple[int, ...]

    @property
    def range_bytes(self) -> int:
        return _product(self.shape) * _byte_width(self.elem_type)


def buffer_plan_for_static_footprint(pipeline: WaveGemmPipelinePlan, elem_type: Optional[str],
                                     shape: Optional[Tuple[int, ...]]) -> Optional[WaveBufferPlan]:
    if not pipeline.use_buffer_ops or elem_type is None or shape is None:
        return None
    if elem_type not in {"f16", "bf16", "f32"}:
        return None
    if not shape or any(dim <= 0 for dim in shape):
        return None
    return WaveBufferPlan(elem_type=elem_type, shape=shape)

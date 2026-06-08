from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Optional, Sequence, Tuple

BUFFER_RANGE_ATTR = "waveamd.buffer.range_bytes"
WAVE_GEMM_STAGE_ATTR = "waveamd.gemm.stage"
WAVE_GEMM_LDS_SLOT_ATTR = "waveamd.gemm.lds_slot"
WAVE_GEMM_LDS_BYTES_ATTR = "waveamd.gemm.lds_bytes"


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
    warp_size: int
    use_lds_staging: bool = True
    fragment_registers: int = 8


@dataclass(frozen=True)
class WaveBufferPlan:
    elem_type: str
    shape: Tuple[int, ...]

    @property
    def range_bytes(self) -> int:
        return _product(self.shape) * _byte_width(self.elem_type)


@dataclass(frozen=True)
class WaveGemmStage:
    k_step: int
    pipeline_stage: int
    a_lds_slot: int
    b_lds_slot: int


@dataclass(frozen=True)
class WaveLdsPlan:
    slots_per_wave: int
    dwords_per_slot: int
    num_warps: int

    @property
    def bytes(self) -> int:
        return self.slots_per_wave * self.dwords_per_slot * self.num_warps * 4

    def slot_offset_dwords(self, slot: int, wave_id, lane, register_count: int):
        slots_per_wave = self.slots_per_wave
        return wave_id * (slots_per_wave * self.dwords_per_slot) + slot * self.dwords_per_slot + lane * register_count


@dataclass(frozen=True)
class WaveGemmSchedule:
    m_tiles: int
    n_tiles: int
    k_steps: int
    stages: Tuple[WaveGemmStage, ...]
    lds: Optional[WaveLdsPlan]


def buffer_plan_for_static_footprint(pipeline: WaveGemmPipelinePlan, elem_type: Optional[str],
                                     shape: Optional[Tuple[int, ...]]) -> Optional[WaveBufferPlan]:
    if not pipeline.use_buffer_ops or elem_type is None or shape is None:
        return None
    if elem_type not in {"f16", "bf16", "f32"}:
        return None
    if not shape or any(dim <= 0 for dim in shape):
        return None
    return WaveBufferPlan(elem_type=elem_type, shape=shape)


def schedule_for_dot(pipeline: WaveGemmPipelinePlan, lhs_shape: Optional[Tuple[int, ...]],
                     rhs_shape: Optional[Tuple[int, ...]],
                     result_shape: Optional[Tuple[int, ...]]) -> Optional[WaveGemmSchedule]:
    if lhs_shape is None or rhs_shape is None or result_shape is None:
        return None
    if len(lhs_shape) != 2 or len(rhs_shape) != 2 or len(result_shape) != 2:
        return None
    m, k = lhs_shape
    rhs_k, n = rhs_shape
    if rhs_k != k or result_shape != (m, n):
        return None
    if m % 16 != 0 or n % 16 != 0 or k % 16 != 0:
        return None
    k_steps = k // 16
    num_stages = max(1, pipeline.num_stages)
    stages = tuple(
        WaveGemmStage(k_step=step, pipeline_stage=step % num_stages, a_lds_slot=0, b_lds_slot=1)
        for step in range(k_steps))
    lds = None
    if pipeline.use_lds_staging:
        lds = WaveLdsPlan(
            slots_per_wave=2,
            dwords_per_slot=pipeline.warp_size * pipeline.fragment_registers,
            num_warps=pipeline.num_warps,
        )
    return WaveGemmSchedule(m_tiles=m // 16, n_tiles=n // 16, k_steps=k_steps, stages=stages, lds=lds)

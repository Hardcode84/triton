from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

from triton._C.libtriton import ir, passes


class PassReuse(str, Enum):
    REUSE_AS_IS = "reuse-as-is"
    REUSE_WITH_CONSTRAINTS = "reuse-with-constraints"
    ADAPT = "adapt"
    AVOID = "avoid"


@dataclass(frozen=True)
class WavePipelinePass:
    name: str
    level: str
    reuse: PassReuse
    purpose: str
    reason: str


@dataclass(frozen=True)
class WaveRunnablePass(WavePipelinePass):
    add_to_pass_manager: Callable[[object], None]


def wave_ttir_cleanup_plan() -> Sequence[WaveRunnablePass]:
    """Executable TTIR cleanup that is safe before Wave-owned lowering."""

    return (
        WaveRunnablePass(
            name="common.inliner",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Inline small helper functions before kernel-local cleanup.",
            reason="Generic MLIR cleanup; does not commit to GPU layout or execution mapping.",
            add_to_pass_manager=passes.common.add_inliner,
        ),
        WaveRunnablePass(
            name="common.canonicalizer",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Apply generic canonicalization before Triton-specific combines.",
            reason="Generic IR cleanup.",
            add_to_pass_manager=passes.common.add_canonicalizer,
        ),
        WaveRunnablePass(
            name="ttir.combine",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Combine Triton TTIR ops into simpler canonical forms.",
            reason="TTIR-local cleanup; does not assign SIMT layouts.",
            add_to_pass_manager=passes.ttir.add_combine,
        ),
        WaveRunnablePass(
            name="ttir.reorder_broadcast",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Move broadcasts into a form expected by later TTIR lowering.",
            reason="TTIR-local canonicalization.",
            add_to_pass_manager=passes.ttir.add_reorder_broadcast,
        ),
        WaveRunnablePass(
            name="common.cse",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Remove duplicate expressions before LICM and unrolling.",
            reason="Generic IR cleanup.",
            add_to_pass_manager=passes.common.add_cse,
        ),
        WaveRunnablePass(
            name="ttir.triton_licm",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Hoist loop-invariant Triton ops while preserving TTIR semantics.",
            reason="TTIR/SCF optimization; already used safely by the Wave backend.",
            add_to_pass_manager=passes.ttir.add_triton_licm,
        ),
        WaveRunnablePass(
            name="common.symbol_dce",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Remove unused symbols before final TTIR shape normalization.",
            reason="Generic IR cleanup.",
            add_to_pass_manager=passes.common.add_symbol_dce,
        ),
        WaveRunnablePass(
            name="ttir.loop_unroll",
            level="ttir",
            reuse=PassReuse.REUSE_AS_IS,
            purpose="Materialize static TTIR loop bodies where Triton expects unrolling.",
            reason="TTIR-level transform that does not impose SIMT layouts.",
            add_to_pass_manager=passes.ttir.add_loop_unroll,
        ),
    )


def add_wave_ttir_cleanup_passes(pm) -> None:
    for stage in wave_ttir_cleanup_plan():
        stage.add_to_pass_manager(pm)


def run_wave_ttir_pipeline(mod, label: str = "wave_amd_make_ttir"):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    add_wave_ttir_cleanup_passes(pm)
    pm.run(mod, label)
    return mod


def wave_ttgir_reuse_plan() -> Sequence[WavePipelinePass]:
    """Curated audit of AMD TTGIR passes for a future Wave TTGIR pipeline."""

    return (
        WavePipelinePass(
            name="ttir.convert_to_ttgpuir",
            level="ttir-to-ttgir",
            reuse=PassReuse.ADAPT,
            purpose="Create the TTGIR boundary and assign initial GPU layouts.",
            reason="The existing pass assigns SIMT layouts; Wave needs a subgroup-aware layout contract.",
        ),
        WavePipelinePass(
            name="ttgpuir.remove_layout_conversions",
            level="ttgir",
            reuse=PassReuse.REUSE_WITH_CONSTRAINTS,
            purpose="Remove redundant TTGIR layout conversions.",
            reason="Safe only after Wave defines which layout conversions must remain visible to lowering.",
        ),
        WavePipelinePass(
            name="ttgpuir.fuse_nested_loops",
            level="ttgir",
            reuse=PassReuse.REUSE_WITH_CONSTRAINTS,
            purpose="Normalize loop structure before software pipeline planning.",
            reason="Useful if Wave scheduling consumes the same loop shape.",
        ),
        WavePipelinePass(
            name="amd.accelerate_matmul",
            level="ttgir",
            reuse=PassReuse.ADAPT,
            purpose="Choose matrix-core-capable dot shapes and operand roles.",
            reason="Dot legality and intrinsic selection are useful, but AMD layout construction is SIMT-specific.",
        ),
        WavePipelinePass(
            name="amd.optimize_dot_operands",
            level="ttgir",
            reuse=PassReuse.ADAPT,
            purpose="Prepare dot operands for shared-memory reuse and matrix instructions.",
            reason="The analysis is useful; LDS layout and operand encodings must be Wave-owned.",
        ),
        WavePipelinePass(
            name="amd.schedule_loops",
            level="ttgir",
            reuse=PassReuse.ADAPT,
            purpose="Build load/use schedules for software pipelining.",
            reason="Schedule representation is useful, but LDS bypass/coalescing checks rely on warp layouts.",
        ),
        WavePipelinePass(
            name="amd.pipeline",
            level="ttgir",
            reuse=PassReuse.ADAPT,
            purpose="Expand scheduled loops into pipelined memory/compute stages.",
            reason="Wave needs its own stream ops, wait model, LDS layout, and predicate lowering.",
        ),
        WavePipelinePass(
            name="amd.canonicalize_pointers",
            level="ttgir",
            reuse=PassReuse.ADAPT,
            purpose="Split pointer bases from vector offsets for range analysis.",
            reason="Pointer algebra is reusable, but current output is staged for AMD raw buffer ops.",
        ),
        WavePipelinePass(
            name="amd.convert_to_buffer_ops",
            level="ttgir",
            reuse=PassReuse.AVOID,
            purpose="Materialize AMDGPU raw buffer operations.",
            reason="Commits to AMDGPU dialect buffer ops instead of Wave buffer descriptors.",
        ),
        WavePipelinePass(
            name="amd.optimize_descriptor_encoding",
            level="ttgir",
            reuse=PassReuse.AVOID,
            purpose="Assign AMD descriptor/TDM encodings.",
            reason="Descriptor and shared-memory layouts should be chosen by Wave-specific passes.",
        ),
        WavePipelinePass(
            name="amd.convert_to_tensor_ops",
            level="ttgir",
            reuse=PassReuse.AVOID,
            purpose="Lower descriptors to AMD tensor/memory ops.",
            reason="Too close to AMD SIMT/LLVM lowering for the Wave path.",
        ),
        WavePipelinePass(
            name="amd.block_pingpong",
            level="ttgir",
            reuse=PassReuse.AVOID,
            purpose="Apply AMD warp/SIMD ping-pong scheduling.",
            reason="Wave machine scheduling should own subgroup scheduling policy.",
        ),
    )


def ttgir_passes_by_reuse(reuse: Optional[PassReuse] = None) -> Sequence[WavePipelinePass]:
    plan = wave_ttgir_reuse_plan()
    if reuse is None:
        return plan
    return tuple(stage for stage in plan if stage.reuse == reuse)

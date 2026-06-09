#ifndef WAVE_AMD_BACKEND_PASSES
#define WAVE_AMD_BACKEND_PASSES

include "mlir/Pass/PassBase.td"

def TritonWaveAMDConvertToTTGPUIR
    : Pass<"tritonwaveamd-convert-to-ttgpuir", "mlir::ModuleOp"> {
  let summary = "Convert Triton to TritonGPU for the Wave AMD backend";

  let description = [{
    Wave-owned entry point for the Triton-to-TritonGPU conversion. It reuses the
    base conversion machinery so the Wave pipeline no longer calls
    `convert-triton-to-tritongpu` directly, giving Wave a seam it can later
    diverge to emit Wave-native layout contracts instead of the default blocked
    encodings.

    Behavior is currently identical to `convert-triton-to-tritongpu`: tensors
    without an encoding get the default blocked layout parameterized by the warp
    configuration, and the module gets the `ttg.num-warps`,
    `ttg.threads-per-warp`, `ttg.num-ctas`, and `ttg.target` attributes.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];

  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"\"",
           "the GPU target, e.g., hip:gfx942">,
    Option<"numWarps", "num-warps", "int32_t", /*default=*/"4",
           "number of warps">,
    Option<"threadsPerWarp", "threads-per-warp", "int32_t", /*default=*/"32",
           "number of threads per warp">,
    Option<"numCTAs", "num-ctas", "int32_t", /*default=*/"1",
           "number of ctas in a cga">,
    Option<"enableSourceRemat", "enable-source-remat", "bool",
           /*default=*/"false", "enable trivial source rematerialization">
  ];
}

def TritonWaveAMDLegalizeDots
    : Pass<"tritonwaveamd-legalize-dots", "mlir::ModuleOp"> {
  let summary = "Legalize TTGIR dots against Wave AMD architecture policy";

  let description = [{
    Wave TTGIR GEMM preparation step that owns dot architecture policy so the
    Python TTGIR-to-Wave bridge never has to parse TTGIR type strings.

    For every `tt.dot` whose result carries an AMD WMMA matrix-core encoding the
    pass attaches typed preparation attributes:
      - `waveamd.dot.instr_kind`: the selected matrix instruction, e.g.
        `wmma.f32.16x16x16.f16`.
      - `waveamd.dot.a_op_idx` / `waveamd.dot.b_op_idx`: explicit operand roles
        recovered from the dot-operand encodings.

    AMD MFMA encodings are rejected with a diagnostic that points at the
    offending op, mirroring the temporary policy currently embedded in
    `lowering.py`. Dots that do not carry a matrix-core encoding are left
    untouched so the pass is safe to run anywhere in the pipeline.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TritonWaveAMDPlanGemmSchedule
    : Pass<"tritonwaveamd-plan-gemm-schedule", "mlir::ModuleOp"> {
  let summary = "Plan Wave AMD GEMM tiling and LDS staging in TTGIR";

  let description = [{
    Wave TTGIR GEMM preparation step that owns the GEMM schedule so the Python
    bridge consumes it mechanically instead of recomputing it.

    For every matrix-core `tt.dot` it attaches the tiling
    (`waveamd.gemm.{m_tiles,n_tiles,k_steps}`) and the per-wave LDS staging plan
    (`waveamd.gemm.lds_{slots_per_wave,dwords_per_slot,bytes}`) derived from the
    module warp configuration. Non matrix-core or non-static dots are left
    unannotated.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];

  let options = [
    Option<"fragmentRegisters", "fragment-registers", "int64_t", /*default=*/"8",
           "Dwords per lane staged through LDS for one dot operand fragment.">
  ];
}

def TritonWaveAMDPlanBufferDescriptors
    : Pass<"tritonwaveamd-plan-buffer-descriptors", "mlir::ModuleOp"> {
  let summary = "Plan Wave AMD buffer descriptor ranges for static footprints";

  let description = [{
    Wave TTGIR GEMM preparation step that decides buffer descriptor ranges in
    TTGIR instead of a post-bridge Wave-dialect materialization pass.

    For every `tt.load` / `tt.store` over a statically-shaped tensor of a
    supported element type, the pass attaches `waveamd.buffer.range_bytes` with
    the footprint size in bytes (product of the static shape times the element
    byte width). The bridge then emits a Wave buffer descriptor mechanically
    from this attribute. Dynamic footprints and unsupported element types are
    left unannotated.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif // WAVE_AMD_BACKEND_PASSES

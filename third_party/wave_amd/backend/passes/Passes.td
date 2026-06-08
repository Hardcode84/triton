#ifndef WAVE_AMD_BACKEND_PASSES
#define WAVE_AMD_BACKEND_PASSES

include "mlir/Pass/PassBase.td"

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

#endif // WAVE_AMD_BACKEND_PASSES

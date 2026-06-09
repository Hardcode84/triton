// RUN: triton-opt %s -split-input-file --tritonwaveamd-legalize-dots --verify-diagnostics | FileCheck %s

// WMMA dots are legalized with a typed dot plan: explicit operand roles and the
// selected matrix instruction kind, so the Python bridge never parses TTGIR
// type strings.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0]]}, instrShape = [16, 16, 32]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wmma_dot
  tt.func @wmma_dot(%a: tensor<16x16xf16, #dot_a>, %b: tensor<16x16xf16, #dot_b>, %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma> {
    // CHECK: tt.dot
    // CHECK-SAME: waveamd.dot.a_op_idx = 0 : i32
    // CHECK-SAME: waveamd.dot.b_op_idx = 1 : i32
    // CHECK-SAME: waveamd.dot.instr_kind = "wmma.f32.16x16x32.f16"
    %d = tt.dot %a, %b, %c : tensor<16x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<16x16xf32, #mma>
    tt.return %d : tensor<16x16xf32, #mma>
  }
}

// -----

// Non-matrix-core dots are left untouched; the pass adds no dot plan.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @blocked_dot
  tt.func @blocked_dot(%a: tensor<16x16xf16, #dot_a>, %b: tensor<16x16xf16, #dot_b>, %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    // CHECK: tt.dot
    // CHECK-NOT: waveamd.dot
    %d = tt.dot %a, %b, %c : tensor<16x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<16x16xf32, #blocked>
    tt.return %d : tensor<16x16xf32, #blocked>
  }
}

// -----

// AMD MFMA encodings are rejected in preparation with a diagnostic that points
// at the offending op, not the later Wave bridge.

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx942"} {
  tt.func @mfma_dot(%a: tensor<16x16xf16, #dot_a>, %b: tensor<16x16xf16, #dot_b>, %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma> {
    // expected-error @+1 {{wave_amd backend does not support AMD MFMA encodings yet}}
    %d = tt.dot %a, %b, %c : tensor<16x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<16x16xf32, #mma>
    tt.return %d : tensor<16x16xf32, #mma>
  }
}

// -----

// Operand convert_layouts carry the role so the bridge reads it from a prepared
// attr instead of parsing the dot_op encoding.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wmma_dot_cvt
  tt.func @wmma_dot_cvt(%a: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %b: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %0 = ttg.convert_layout %c : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #mma>
    // CHECK: ttg.convert_layout %arg0 {waveamd.dot.role = 0 : i32}
    %1 = ttg.convert_layout %a : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK: ttg.convert_layout %arg1 {waveamd.dot.role = 1 : i32}
    %2 = ttg.convert_layout %b : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %3 = tt.dot %1, %2, %0 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    %4 = ttg.convert_layout %3 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    tt.return %4 : tensor<16x16xf32, #blocked>
  }
}

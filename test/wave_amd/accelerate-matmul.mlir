// RUN: triton-opt %s -split-input-file --tritonwaveamd-accelerate-matmul="gfx-arch=gfx1100 matrix-instruction-size=0 kpack=1" | FileCheck %s

// The Wave-owned dot legalization reuses the base AMD accelerate-matmul, so a
// blocked f16 dot on RDNA3 is rewritten to an AMD WMMA v1 result with explicit
// operand-role dot encodings (kWidth = 16). Output is identical to
// tritonamdgpu-accelerate-matmul; this guards that contract.

// CHECK: #mma = #ttg.amd_wmma<{version = 1, isTranspose = true
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @gemm
  tt.func public @gemm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.splat %a : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %2 = tt.splat %b : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %2 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %4 = ttg.convert_layout %1 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %5 = ttg.convert_layout %3 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: tt.dot
    // CHECK-SAME: parent = #mma, kWidth = 16
    // CHECK-SAME: -> tensor<32x32xf32, #mma>
    %6 = tt.dot %4, %5, %cst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf32, #blocked>
    %7 = tt.splat %c : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %7, %6 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

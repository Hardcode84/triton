// RUN: triton-opt %s -split-input-file --tritonwaveamd-plan-gemm-schedule | FileCheck %s

// A matrix-core dot gets the GEMM tiling and per-wave LDS staging plan attached
// as typed attrs. num_warps=4, threads-per-warp=32, fragment-registers=8:
//   lds_dwords_per_slot = 32 * 8 = 256
//   lds_bytes = slots_per_wave(2) * 256 * num_warps(4) * 4 = 8192

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0]]}, instrShape = [16, 16, 32]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wmma_dot
  tt.func @wmma_dot(%a: tensor<16x16xf16, #dot_a>, %b: tensor<16x16xf16, #dot_b>, %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma> {
    // CHECK: tt.dot
    // CHECK-SAME: waveamd.gemm.k_steps = 1 : i32
    // CHECK-SAME: waveamd.gemm.lds_bytes = 8192 : i32
    // CHECK-SAME: waveamd.gemm.lds_dwords_per_slot = 256 : i32
    // CHECK-SAME: waveamd.gemm.lds_slots_per_wave = 2 : i32
    // CHECK-SAME: waveamd.gemm.m_tiles = 1 : i32
    // CHECK-SAME: waveamd.gemm.n_tiles = 1 : i32
    %d = tt.dot %a, %b, %c : tensor<16x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<16x16xf32, #mma>
    tt.return %d : tensor<16x16xf32, #mma>
  }
}

// -----

// Non-matrix-core dots are left unplanned.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @blocked_dot
  tt.func @blocked_dot(%a: tensor<16x16xf16, #dot_a>, %b: tensor<16x16xf16, #dot_b>, %c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    // CHECK: tt.dot
    // CHECK-NOT: waveamd.gemm
    %d = tt.dot %a, %b, %c : tensor<16x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<16x16xf32, #blocked>
    tt.return %d : tensor<16x16xf32, #blocked>
  }
}

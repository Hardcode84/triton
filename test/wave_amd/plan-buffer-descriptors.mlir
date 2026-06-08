// RUN: triton-opt %s -split-input-file --tritonwaveamd-plan-buffer-descriptors | FileCheck %s

// Static-footprint loads and stores of supported element types get a buffer
// descriptor range in bytes (product of the static shape times the element
// byte width): 16 * 32 * 2 = 1024 for f16, 8 * 8 * 4 = 256 for f32.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @static_footprints
  tt.func @static_footprints(%p: tensor<16x32x!tt.ptr<f16>, #blocked>, %v: tensor<16x32xf16, #blocked>, %q: tensor<8x8x!tt.ptr<f32>, #blocked>) {
    // CHECK: tt.load
    // CHECK-SAME: waveamd.buffer.range_bytes = 1024 : i32
    %a = tt.load %p : tensor<16x32x!tt.ptr<f16>, #blocked>
    // CHECK: tt.store
    // CHECK-SAME: waveamd.buffer.range_bytes = 1024 : i32
    tt.store %p, %v : tensor<16x32x!tt.ptr<f16>, #blocked>
    // CHECK: tt.load
    // CHECK-SAME: waveamd.buffer.range_bytes = 256 : i32
    %b = tt.load %q : tensor<8x8x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Scalar loads have no tensor footprint and unsupported element types (i8) are
// not planned; neither gets a range attribute.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @unplanned
  tt.func @unplanned(%s: !tt.ptr<f16>, %p: tensor<16x16x!tt.ptr<i8>, #blocked>) {
    // CHECK: tt.load
    // CHECK-NOT: waveamd.buffer.range_bytes
    %a = tt.load %s : !tt.ptr<f16>
    // CHECK: tt.load
    // CHECK-NOT: waveamd.buffer.range_bytes
    %b = tt.load %p : tensor<16x16x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

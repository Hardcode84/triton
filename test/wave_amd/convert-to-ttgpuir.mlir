// RUN: triton-opt %s -split-input-file --tritonwaveamd-convert-to-ttgpuir="target=hip:gfx1100 num-warps=1 threads-per-warp=32 num-ctas=1" | FileCheck %s

// The Wave-owned conversion reuses the base Triton-to-TritonGPU machinery, so it
// assigns the default blocked encodings parameterized by the warp configuration
// and stamps the module with the ttg target/warp attributes. Its output is
// identical to convert-triton-to-tritongpu; this guards that contract.

// CHECK: #blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: module attributes {
// CHECK-SAME: "ttg.num-ctas" = 1 : i32
// CHECK-SAME: "ttg.num-warps" = 1 : i32
// CHECK-SAME: ttg.target = "hip:gfx1100"
// CHECK-SAME: "ttg.threads-per-warp" = 32 : i32
module {
  // CHECK-LABEL: @add_kernel
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
    // CHECK: tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    // CHECK: tt.load %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<32x32x!tt.ptr<f32>>
    %2 = arith.addf %1, %cst : tensor<32x32xf32>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}

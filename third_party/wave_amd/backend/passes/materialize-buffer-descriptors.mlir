// RUN: wave-opt --split-input-file --waveamd-materialize-buffer-descriptors %s | FileCheck %s
//
// Staged outside the Wave submodule until the pass is wired into the Wave
// transform library.

// CHECK-LABEL: func.func @materialize_cached_descriptors
func.func @materialize_cached_descriptors(%a: !wave.ptr<#wave.global, f16>,
                                          %b: !wave.ptr<#wave.global, f16>) attributes {wave.kernel} {
  %lane = wave.lane_id : !wave.simd<i32, 32>
  %off = wave.index_expr <"lane"> ["lane"](%lane)
      : (!wave.simd<i32, 32>) -> !wave.simd<index, 32>

  // CHECK: %[[A_RANGE:.*]] = arith.constant 512 : i32
  // CHECK: %[[A_BUFFER:.*]] = waveamd.make_buffer %arg0, %[[A_RANGE]] : !wave.ptr<#wave.global, f16>, i32 -> !wave.ptr<#waveamd.buffer, f16>
  // CHECK: %[[A_PTR0:.*]] = wave.ptr_add %[[A_BUFFER]], %{{.*}} : !wave.ptr<#waveamd.buffer, f16>, !wave.simd<index, 32> -> !wave.simd<!wave.ptr<#waveamd.buffer, f16>, 32>
  %a_ptr0 = wave.ptr_add %a, %off {waveamd.buffer.range_bytes = 512 : i32}
      : !wave.ptr<#wave.global, f16>, !wave.simd<index, 32>
      -> !wave.simd<!wave.ptr<#wave.global, f16>, 32>
  %a0, %a_tok0 = wave.load %a_ptr0
      : (!wave.simd<!wave.ptr<#wave.global, f16>, 32>)
      -> (!wave.simd<f16, 32>, !wave.mem.token)

  // CHECK-NOT: waveamd.make_buffer %arg0
  // CHECK: %[[A_PTR1:.*]] = wave.ptr_add %[[A_BUFFER]], %{{.*}} : !wave.ptr<#waveamd.buffer, f16>, !wave.simd<index, 32> -> !wave.simd<!wave.ptr<#waveamd.buffer, f16>, 32>
  %a_ptr1 = wave.ptr_add %a, %off {waveamd.buffer.range_bytes = 512 : i32}
      : !wave.ptr<#wave.global, f16>, !wave.simd<index, 32>
      -> !wave.simd<!wave.ptr<#wave.global, f16>, 32>
  %a1, %a_tok1 = wave.load %a_ptr1 after %a_tok0
      : (!wave.simd<!wave.ptr<#wave.global, f16>, 32>, !wave.mem.token)
      -> (!wave.simd<f16, 32>, !wave.mem.token)

  // CHECK: %[[B_RANGE:.*]] = arith.constant 1024 : i32
  // CHECK: %[[B_BUFFER:.*]] = waveamd.make_buffer %arg1, %[[B_RANGE]] : !wave.ptr<#wave.global, f16>, i32 -> !wave.ptr<#waveamd.buffer, f16>
  // CHECK: wave.ptr_add %[[B_BUFFER]], %{{.*}} : !wave.ptr<#waveamd.buffer, f16>, !wave.simd<index, 32> -> !wave.simd<!wave.ptr<#waveamd.buffer, f16>, 32>
  %b_ptr = wave.ptr_add %b, %off {waveamd.buffer.range_bytes = 1024 : i32}
      : !wave.ptr<#wave.global, f16>, !wave.simd<index, 32>
      -> !wave.simd<!wave.ptr<#wave.global, f16>, 32>
  %b0, %b_tok0 = wave.load %b_ptr after %a_tok1
      : (!wave.simd<!wave.ptr<#wave.global, f16>, 32>, !wave.mem.token)
      -> (!wave.simd<f16, 32>, !wave.mem.token)

  // CHECK-NOT: waveamd.buffer.range_bytes
  return
}

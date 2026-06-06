import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
libtriton = pytest.importorskip("triton._C.libtriton")
compiler_api = pytest.importorskip("triton.backends.compiler")
triton_compiler = pytest.importorskip("triton.compiler.compiler")
driver_api = pytest.importorskip("triton.backends.driver")
wave_compiler = pytest.importorskip("triton.backends.wave_amd.compiler")
wave_emission = pytest.importorskip("triton.backends.wave_amd.emission")
wave_lowering = pytest.importorskip("triton.backends.wave_amd.lowering")
wave_pipeline = pytest.importorskip("triton.backends.wave_amd.pipeline")
wave_driver = pytest.importorskip("triton.backends.wave_amd.driver")
hip_driver = pytest.importorskip("triton.backends.amd.driver")

GPUTarget = compiler_api.GPUTarget
Language = compiler_api.Language
DriverBase = driver_api.DriverBase
WaveAMDBackend = wave_compiler.WaveAMDBackend
ir = libtriton.ir

ELEMENTWISE_ADD_TTIR = """
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %b_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %a_ptr = tt.addptr %a_base, %lane : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %b_ptr = tt.addptr %b_base, %lane : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %c_ptr = tt.addptr %c_base, %lane : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %a = tt.load %a_ptr : tensor<32x!tt.ptr<f32>>
    %b = tt.load %b_ptr : tensor<32x!tt.ptr<f32>>
    %sum = arith.addf %a, %b : tensor<32xf32>
    tt.store %c_ptr, %sum : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
"""

MULTI_REGISTER_ADD_TTIR = """
module {
  tt.func public @wide_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                  %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                  %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %lane = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %b_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %a_ptr = tt.addptr %a_base, %lane : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %b_ptr = tt.addptr %b_base, %lane : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %c_ptr = tt.addptr %c_base, %lane : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %a = tt.load %a_ptr : tensor<64x!tt.ptr<f32>>
    %b = tt.load %b_ptr : tensor<64x!tt.ptr<f32>>
    %sum = arith.addf %a, %b : tensor<64xf32>
    tt.store %c_ptr, %sum : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""

TWO_D_STORE_TTIR = """
module {
  tt.func public @store_2d_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32 = arith.constant 32 : i32
    %one = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %rows_b = tt.broadcast %rows_2d : tensor<32x1xi32> -> tensor<32x32xi32>
    %cols_b = tt.broadcast %cols_2d : tensor<1x32xi32> -> tensor<32x32xi32>
    %stride = tt.splat %c32 : i32 -> tensor<32x32xi32>
    %row_offsets = arith.muli %rows_b, %stride : tensor<32x32xi32>
    %offs = arith.addi %row_offsets, %cols_b : tensor<32x32xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %offs : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %ptrs, %one : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}
"""

TWO_D_MULTI_WARP_STORE_TTIR = """
module {
  tt.func public @store_2d_multi_warp_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c64 = arith.constant 64 : i32
    %one = arith.constant dense<1.000000e+00> : tensor<32x64xf32>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %rows_b = tt.broadcast %rows_2d : tensor<32x1xi32> -> tensor<32x64xi32>
    %cols_b = tt.broadcast %cols_2d : tensor<1x64xi32> -> tensor<32x64xi32>
    %stride = tt.splat %c64 : i32 -> tensor<32x64xi32>
    %row_offsets = arith.muli %rows_b, %stride : tensor<32x64xi32>
    %offs = arith.addi %row_offsets, %cols_b : tensor<32x64xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %offs : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    tt.store %ptrs, %one : tensor<32x64x!tt.ptr<f32>>
    tt.return
  }
}
"""

TWO_D_MULTI_CTA_STORE_TTIR = """
module {
  tt.func public @store_2d_multi_cta_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c64 = arith.constant 64 : i32
    %one = arith.constant dense<1.000000e+00> : tensor<32x64xf32>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %rows_b = tt.broadcast %rows_2d : tensor<32x1xi32> -> tensor<32x64xi32>
    %cols_b = tt.broadcast %cols_2d : tensor<1x64xi32> -> tensor<32x64xi32>
    %stride = tt.splat %c64 : i32 -> tensor<32x64xi32>
    %row_offsets = arith.muli %rows_b, %stride : tensor<32x64xi32>
    %offs = arith.addi %row_offsets, %cols_b : tensor<32x64xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %offs : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    tt.store %ptrs, %one : tensor<32x64x!tt.ptr<f32>>
    tt.return
  }
}
"""

MULTI_CTA_PROGRAM_ID_STORE_TTIR = """
module {
  tt.func public @store_multi_cta_pid_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c64 = arith.constant 64 : i32
    %one = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %pid = tt.get_program_id x : i32
    %block = arith.muli %pid, %c64 : i32
    %block_vec = tt.splat %block : i32 -> tensor<64xi32>
    %lane = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offs = arith.addi %block_vec, %lane : tensor<64xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %ptrs, %one : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""


def _dot_matmul_ttir(m: int, n: int, k: int, name: str) -> str:
    return f"""
module {{
  tt.func public @{name}(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}},
                        %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}},
                        %arg2: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
    %cn = arith.constant {n} : i32
    %ck = arith.constant {k} : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<{m}x{n}xf32>
    %rows = tt.make_range {{end = {m} : i32, start = 0 : i32}} : tensor<{m}xi32>
    %cols = tt.make_range {{end = {n} : i32, start = 0 : i32}} : tensor<{n}xi32>
    %ks = tt.make_range {{end = {k} : i32, start = 0 : i32}} : tensor<{k}xi32>
    %rows_a = tt.expand_dims %rows {{axis = 1 : i32}} : tensor<{m}xi32> -> tensor<{m}x1xi32>
    %ks_a = tt.expand_dims %ks {{axis = 0 : i32}} : tensor<{k}xi32> -> tensor<1x{k}xi32>
    %rows_a_b = tt.broadcast %rows_a : tensor<{m}x1xi32> -> tensor<{m}x{k}xi32>
    %ks_a_b = tt.broadcast %ks_a : tensor<1x{k}xi32> -> tensor<{m}x{k}xi32>
    %a_stride = tt.splat %ck : i32 -> tensor<{m}x{k}xi32>
    %a_row_offsets = arith.muli %rows_a_b, %a_stride : tensor<{m}x{k}xi32>
    %a_offsets = arith.addi %a_row_offsets, %ks_a_b : tensor<{m}x{k}xi32>
    %ks_b = tt.expand_dims %ks {{axis = 1 : i32}} : tensor<{k}xi32> -> tensor<{k}x1xi32>
    %cols_b = tt.expand_dims %cols {{axis = 0 : i32}} : tensor<{n}xi32> -> tensor<1x{n}xi32>
    %ks_b_b = tt.broadcast %ks_b : tensor<{k}x1xi32> -> tensor<{k}x{n}xi32>
    %cols_b_b = tt.broadcast %cols_b : tensor<1x{n}xi32> -> tensor<{k}x{n}xi32>
    %b_stride = tt.splat %ck : i32 -> tensor<{k}x{n}xi32>
    %b_col_offsets = arith.muli %cols_b_b, %b_stride : tensor<{k}x{n}xi32>
    %b_offsets = arith.addi %b_col_offsets, %ks_b_b : tensor<{k}x{n}xi32>
    %rows_c = tt.expand_dims %rows {{axis = 1 : i32}} : tensor<{m}xi32> -> tensor<{m}x1xi32>
    %cols_c = tt.expand_dims %cols {{axis = 0 : i32}} : tensor<{n}xi32> -> tensor<1x{n}xi32>
    %rows_c_b = tt.broadcast %rows_c : tensor<{m}x1xi32> -> tensor<{m}x{n}xi32>
    %cols_c_b = tt.broadcast %cols_c : tensor<1x{n}xi32> -> tensor<{m}x{n}xi32>
    %c_stride = tt.splat %cn : i32 -> tensor<{m}x{n}xi32>
    %c_row_offsets = arith.muli %rows_c_b, %c_stride : tensor<{m}x{n}xi32>
    %c_offsets = arith.addi %c_row_offsets, %cols_c_b : tensor<{m}x{n}xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{m}x{k}x!tt.ptr<f16>>
    %b_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{k}x{n}x!tt.ptr<f16>>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<{m}x{n}x!tt.ptr<f32>>
    %a_ptrs = tt.addptr %a_base, %a_offsets : tensor<{m}x{k}x!tt.ptr<f16>>, tensor<{m}x{k}xi32>
    %b_ptrs = tt.addptr %b_base, %b_offsets : tensor<{k}x{n}x!tt.ptr<f16>>, tensor<{k}x{n}xi32>
    %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<{m}x{n}x!tt.ptr<f32>>, tensor<{m}x{n}xi32>
    %a = tt.load %a_ptrs : tensor<{m}x{k}x!tt.ptr<f16>>
    %b = tt.load %b_ptrs : tensor<{k}x{n}x!tt.ptr<f16>>
    %acc = tt.dot %a, %b, %zero : tensor<{m}x{k}xf16> * tensor<{k}x{n}xf16> -> tensor<{m}x{n}xf32>
    tt.store %c_ptrs, %acc : tensor<{m}x{n}x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


DOT_MATMUL_TTIR = _dot_matmul_ttir(16, 16, 16, "dot_kernel")
DOT_MATMUL_K32_TTIR = _dot_matmul_ttir(16, 16, 32, "dot_k32_kernel")
DOT_MATMUL_K8_TTIR = _dot_matmul_ttir(16, 16, 8, "dot_k8_kernel")
DOT_MATMUL_32X32_TTIR = _dot_matmul_ttir(32, 32, 16, "dot_32x32_kernel")
DOT_MATMUL_32X32_K32_TTIR = _dot_matmul_ttir(32, 32, 32, "dot_32x32_k32_kernel")
DOT_MATMUL_M8_TTIR = _dot_matmul_ttir(8, 32, 16, "dot_m8_kernel")
DOT_MATMUL_N8_TTIR = _dot_matmul_ttir(32, 8, 16, "dot_n8_kernel")

REALISTIC_MATMUL_TILE_TTIR = """
module {
  tt.func public @realistic_matmul_tile_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                               %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                               %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %arg3: i32,
                                               %arg4: i32,
                                               %arg5: i32,
                                               %arg6: i32,
                                               %arg7: i32,
                                               %arg8: i32,
                                               %arg9: i32,
                                               %arg10: i32,
                                               %arg11: i32) attributes {noinline = false} {
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16>
    %pid = tt.get_program_id x : i32
    %pid_m = arith.remsi %pid, %c2 : i32
    %pid_n = arith.divsi %pid, %c2 : i32
    %pid_m_block = arith.muli %pid_m, %c16 : i32
    %pid_n_block = arith.muli %pid_n, %c16 : i32
    %pid_m_vec = tt.splat %pid_m_block : i32 -> tensor<16xi32>
    %pid_n_vec = tt.splat %pid_n_block : i32 -> tensor<16xi32>
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %ks = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %offs_m = arith.addi %pid_m_vec, %rows : tensor<16xi32>
    %offs_n = arith.addi %pid_n_vec, %cols : tensor<16xi32>
    %rows_a = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %ks_a = tt.expand_dims %ks {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rows_a_b = tt.broadcast %rows_a : tensor<16x1xi32> -> tensor<16x16xi32>
    %ks_a_b = tt.broadcast %ks_a : tensor<1x16xi32> -> tensor<16x16xi32>
    %stride_am = tt.splat %arg6 : i32 -> tensor<16x16xi32>
    %stride_ak = tt.splat %arg7 : i32 -> tensor<16x16xi32>
    %a_row_offsets = arith.muli %rows_a_b, %stride_am : tensor<16x16xi32>
    %a_k_offsets = arith.muli %ks_a_b, %stride_ak : tensor<16x16xi32>
    %a_offsets = arith.addi %a_row_offsets, %a_k_offsets : tensor<16x16xi32>
    %m_vec = tt.splat %arg3 : i32 -> tensor<16x16xi32>
    %k_vec = tt.splat %arg5 : i32 -> tensor<16x16xi32>
    %a_m_mask = arith.cmpi ult, %rows_a_b, %m_vec : tensor<16x16xi32>
    %a_k_mask = arith.cmpi ult, %ks_a_b, %k_vec : tensor<16x16xi32>
    %a_mask = arith.andi %a_m_mask, %a_k_mask : tensor<16x16xi1>
    %ks_b = tt.expand_dims %ks {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cols_b = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %ks_b_b = tt.broadcast %ks_b : tensor<16x1xi32> -> tensor<16x16xi32>
    %cols_b_b = tt.broadcast %cols_b : tensor<1x16xi32> -> tensor<16x16xi32>
    %stride_bk = tt.splat %arg8 : i32 -> tensor<16x16xi32>
    %stride_bn = tt.splat %arg9 : i32 -> tensor<16x16xi32>
    %b_k_offsets = arith.muli %ks_b_b, %stride_bk : tensor<16x16xi32>
    %b_col_offsets = arith.muli %cols_b_b, %stride_bn : tensor<16x16xi32>
    %b_offsets = arith.addi %b_k_offsets, %b_col_offsets : tensor<16x16xi32>
    %n_vec = tt.splat %arg4 : i32 -> tensor<16x16xi32>
    %b_k_mask = arith.cmpi ult, %ks_b_b, %k_vec : tensor<16x16xi32>
    %b_n_mask = arith.cmpi ult, %cols_b_b, %n_vec : tensor<16x16xi32>
    %b_mask = arith.andi %b_k_mask, %b_n_mask : tensor<16x16xi1>
    %rows_c = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cols_c = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rows_c_b = tt.broadcast %rows_c : tensor<16x1xi32> -> tensor<16x16xi32>
    %cols_c_b = tt.broadcast %cols_c : tensor<1x16xi32> -> tensor<16x16xi32>
    %stride_cm = tt.splat %arg10 : i32 -> tensor<16x16xi32>
    %stride_cn = tt.splat %arg11 : i32 -> tensor<16x16xi32>
    %c_row_offsets = arith.muli %rows_c_b, %stride_cm : tensor<16x16xi32>
    %c_col_offsets = arith.muli %cols_c_b, %stride_cn : tensor<16x16xi32>
    %c_offsets = arith.addi %c_row_offsets, %c_col_offsets : tensor<16x16xi32>
    %c_m_mask = arith.cmpi ult, %rows_c_b, %m_vec : tensor<16x16xi32>
    %c_n_mask = arith.cmpi ult, %cols_c_b, %n_vec : tensor<16x16xi32>
    %c_mask = arith.andi %c_m_mask, %c_n_mask : tensor<16x16xi1>
    %a_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %b_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %a_ptrs = tt.addptr %a_base, %a_offsets : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %b_ptrs = tt.addptr %b_base, %b_offsets : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %a = tt.load %a_ptrs, %a_mask, %other : tensor<16x16x!tt.ptr<f16>>
    %b = tt.load %b_ptrs, %b_mask, %other : tensor<16x16x!tt.ptr<f16>>
    %acc = tt.dot %a, %b, %zero : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    tt.store %c_ptrs, %acc, %c_mask : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
"""

REALISTIC_MATMUL_LOOP_TTIR = """
module {
  tt.func public @realistic_matmul_loop_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                               %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                               %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c16idx = arith.constant 16 : index
    %c32idx = arith.constant 32 : index
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %rows = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %ks = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %rows_a = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %ks_a = tt.expand_dims %ks {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rows_a_b = tt.broadcast %rows_a : tensor<16x1xi32> -> tensor<16x16xi32>
    %ks_a_b = tt.broadcast %ks_a : tensor<1x16xi32> -> tensor<16x16xi32>
    %a_stride = tt.splat %c32 : i32 -> tensor<16x16xi32>
    %a_row_offsets = arith.muli %rows_a_b, %a_stride : tensor<16x16xi32>
    %ks_b = tt.expand_dims %ks {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cols_b = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %ks_b_b = tt.broadcast %ks_b : tensor<16x1xi32> -> tensor<16x16xi32>
    %cols_b_b = tt.broadcast %cols_b : tensor<1x16xi32> -> tensor<16x16xi32>
    %b_stride = tt.splat %c32 : i32 -> tensor<16x16xi32>
    %b_col_offsets = arith.muli %cols_b_b, %b_stride : tensor<16x16xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %b_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %acc = scf.for %i = %c0 to %c32idx step %c16idx iter_args(%iter = %zero) -> (tensor<16x16xf32>) {
      %i_i32 = arith.index_cast %i : index to i32
      %i_a = tt.splat %i_i32 : i32 -> tensor<16x16xi32>
      %a_k_offsets = arith.addi %i_a, %ks_a_b : tensor<16x16xi32>
      %a_offsets = arith.addi %a_row_offsets, %a_k_offsets : tensor<16x16xi32>
      %b_k_offsets = arith.addi %i_a, %ks_b_b : tensor<16x16xi32>
      %b_offsets = arith.addi %b_col_offsets, %b_k_offsets : tensor<16x16xi32>
      %a_ptrs = tt.addptr %a_base, %a_offsets : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      %b_ptrs = tt.addptr %b_base, %b_offsets : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      %a = tt.load %a_ptrs : tensor<16x16x!tt.ptr<f16>>
      %b = tt.load %b_ptrs : tensor<16x16x!tt.ptr<f16>>
      %next = tt.dot %a, %b, %iter : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }
    %rows_c = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %cols_c = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %rows_c_b = tt.broadcast %rows_c : tensor<16x1xi32> -> tensor<16x16xi32>
    %cols_c_b = tt.broadcast %cols_c : tensor<1x16xi32> -> tensor<16x16xi32>
    %c_stride = tt.splat %c16 : i32 -> tensor<16x16xi32>
    %c_row_offsets = arith.muli %rows_c_b, %c_stride : tensor<16x16xi32>
    %c_offsets = arith.addi %c_row_offsets, %cols_c_b : tensor<16x16xi32>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %c_ptrs = tt.addptr %c_base, %c_offsets : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %c_ptrs, %acc : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
"""

BOUNDARY_MATMUL_TILE_TTIR = REALISTIC_MATMUL_TILE_TTIR.replace(
    "realistic_matmul_tile_kernel", "boundary_matmul_tile_kernel").replace(
        """    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16>
    %pid = tt.get_program_id x : i32
    %pid_m = arith.remsi %pid, %c2 : i32
    %pid_n = arith.divsi %pid, %c2 : i32
    %pid_m_block = arith.muli %pid_m, %c16 : i32
    %pid_n_block = arith.muli %pid_n, %c16 : i32
    %pid_m_vec = tt.splat %pid_m_block : i32 -> tensor<16xi32>
    %pid_n_vec = tt.splat %pid_n_block : i32 -> tensor<16xi32>""",
        """    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16>
    %c0_i32 = arith.constant 0 : i32
    %pid_m_vec = tt.splat %c0_i32 : i32 -> tensor<16xi32>
    %pid_n_vec = tt.splat %c0_i32 : i32 -> tensor<16xi32>""")

MASKED_ADD_TTIR = """
module {
  tt.func public @masked_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                    %arg3: i32) attributes {noinline = false} {
    %c32 = arith.constant 32 : i32
    %pid = tt.get_program_id x : i32
    %block = arith.muli %pid, %c32 : i32
    %block_vec = tt.splat %block : i32 -> tensor<32xi32>
    %lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %offs = arith.addi %block_vec, %lane : tensor<32xi32>
    %n_vec = tt.splat %arg3 : i32 -> tensor<32xi32>
    %mask = arith.cmpi ult, %offs, %n_vec : tensor<32xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %b_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %a_ptr = tt.addptr %a_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %b_ptr = tt.addptr %b_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %c_ptr = tt.addptr %c_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %a = tt.load %a_ptr, %mask : tensor<32x!tt.ptr<f32>>
    %b = tt.load %b_ptr, %mask : tensor<32x!tt.ptr<f32>>
    %sum = arith.addf %a, %b : tensor<32xf32>
    tt.store %c_ptr, %sum, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
"""

MASKED_LOAD_OTHER_TTIR = """
module {
  tt.func public @masked_load_other_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                          %arg2: i32) attributes {noinline = false} {
    %c32 = arith.constant 32 : i32
    %other = arith.constant dense<5.000000e+00> : tensor<32xf32>
    %pid = tt.get_program_id x : i32
    %block = arith.muli %pid, %c32 : i32
    %block_vec = tt.splat %block : i32 -> tensor<32xi32>
    %lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %offs = arith.addi %block_vec, %lane : tensor<32xi32>
    %n_vec = tt.splat %arg2 : i32 -> tensor<32xi32>
    %mask = arith.cmpi ult, %offs, %n_vec : tensor<32xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %c_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %a_ptr = tt.addptr %a_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %c_ptr = tt.addptr %c_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %a = tt.load %a_ptr, %mask, %other : tensor<32x!tt.ptr<f32>>
    tt.store %c_ptr, %a : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
"""

MASKED_SUB_MUL_TTIR = """
module {
  tt.func public @masked_sub_mul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %arg3: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %pid = tt.get_program_id x : i32
    %block = arith.muli %pid, %c32 : i32
    %block_vec = tt.splat %block : i32 -> tensor<32xi32>
    %lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %raw_offs = arith.addi %block_vec, %lane : tensor<32xi32>
    %zero_vec = tt.splat %c0 : i32 -> tensor<32xi32>
    %offs = arith.subi %raw_offs, %zero_vec : tensor<32xi32>
    %n_vec = tt.splat %arg3 : i32 -> tensor<32xi32>
    %mask = arith.cmpi ult, %offs, %n_vec : tensor<32xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %b_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %a_ptr = tt.addptr %a_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %b_ptr = tt.addptr %b_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %c_ptr = tt.addptr %c_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %a = tt.load %a_ptr, %mask : tensor<32x!tt.ptr<f32>>
    %b = tt.load %b_ptr, %mask : tensor<32x!tt.ptr<f32>>
    %diff = arith.subf %a, %b : tensor<32xf32>
    %prod = arith.mulf %diff, %b : tensor<32xf32>
    tt.store %c_ptr, %prod, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
"""

MASKED_SELECT_TTIR = """
module {
  tt.func public @masked_select_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg3: i32) attributes {noinline = false} {
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %pid = tt.get_program_id x : i32
    %block = arith.muli %pid, %c32 : i32
    %block_vec = tt.splat %block : i32 -> tensor<32xi32>
    %lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %offs = arith.addi %block_vec, %lane : tensor<32xi32>
    %n_vec = tt.splat %arg3 : i32 -> tensor<32xi32>
    %mask = arith.cmpi ult, %offs, %n_vec : tensor<32xi32>
    %half_vec = tt.splat %c16 : i32 -> tensor<32xi32>
    %choose_a = arith.cmpi ult, %lane, %half_vec : tensor<32xi32>
    %a_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %b_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %c_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %a_ptr = tt.addptr %a_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %b_ptr = tt.addptr %b_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %c_ptr = tt.addptr %c_base, %offs : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %a = tt.load %a_ptr, %mask : tensor<32x!tt.ptr<f32>>
    %b = tt.load %b_ptr, %mask : tensor<32x!tt.ptr<f32>>
    %selected = arith.select %choose_a, %a, %b : tensor<32xi1>, tensor<32xf32>
    tt.store %c_ptr, %selected, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
"""


def _parse_ttir(tmp_path, backend, ttir):
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    path = tmp_path / "kernel.ttir"
    path.write_text(ttir)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def _fake_result_tensor_info(op, result_index):
    if result_index >= op.get_num_results():
        return None
    type_text = str(op.get_result(result_index).get_type())
    if not type_text.startswith("tensor<") or not type_text.endswith(">"):
        return None
    body = type_text[len("tensor<"):-1]
    shape_text, elem_type = body.rsplit("x", 1)
    shape = tuple(int(dim) for dim in shape_text.split("x"))
    if elem_type.startswith("!tt.ptr<") and elem_type.endswith(">"):
        return shape, elem_type[len("!tt.ptr<"):-1], True
    return shape, elem_type, False


def test_wave_amd_backend_skeleton():
    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({})

    assert WaveAMDBackend.supports_target(target)
    assert backend.binary_ext == "hsaco"
    assert backend.get_target_name(options) == "wave_amd:gfx1100"
    assert options.backend_name == "wave_amd"
    assert options.warp_size == 32
    assert backend.pack_metadata(SimpleNamespace(num_warps=4, num_ctas=1)) == (4, 1, 0)

    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    assert list(stages) == ["ttir", "ttgir", "wave", "amdgcn", "hsaco"]

    preview_options = backend.parse_options({"enable_ttgir_preview": True})
    preview_stages = {}
    backend.add_stages(preview_stages, preview_options, Language.TRITON)
    assert list(preview_stages) == ["ttir", "ttgir_preview", "ttgir", "wave", "amdgcn", "hsaco"]


def test_wave_amd_pipeline_records_safe_ttir_cleanup_and_ttgir_reuse_plan():
    assert [stage.name for stage in wave_pipeline.wave_ttir_cleanup_plan()] == [
        "common.inliner",
        "common.canonicalizer",
        "ttir.combine",
        "ttir.reorder_broadcast",
        "common.cse",
        "ttir.triton_licm",
        "common.symbol_dce",
        "ttir.loop_unroll",
    ]

    ttgir_plan = {stage.name: stage.reuse for stage in wave_pipeline.wave_ttgir_reuse_plan()}
    assert ttgir_plan["ttir.convert_to_ttgpuir"] == wave_pipeline.PassReuse.ADAPT
    assert ttgir_plan["amd.accelerate_matmul"] == wave_pipeline.PassReuse.ADAPT
    assert ttgir_plan["amd.schedule_loops"] == wave_pipeline.PassReuse.ADAPT
    assert ttgir_plan["amd.convert_to_buffer_ops"] == wave_pipeline.PassReuse.AVOID
    assert ttgir_plan["amd.block_pingpong"] == wave_pipeline.PassReuse.AVOID
    assert "ttgpuir.remove_layout_conversions" in {
        stage.name
        for stage in wave_pipeline.ttgir_passes_by_reuse(wave_pipeline.PassReuse.REUSE_WITH_CONSTRAINTS)
    }


def test_wave_amd_ttgir_preview_keeps_ttir_as_lowering_input(tmp_path):
    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "enable_ttgir_preview": True})
    module = _parse_ttir(tmp_path, backend, ELEMENTWISE_ADD_TTIR)
    metadata = {}

    returned = backend.make_ttgir_preview(module, metadata, options)

    assert returned is module
    assert "wave_ttgir_preview" in metadata
    assert "#ttg." in metadata["wave_ttgir_preview"]
    assert "tt.func" in str(module)


def _ttgir_preview_text(tmp_path, ttir, options_dict=None):
    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    opts = {"num_warps": 1, "enable_ttgir_preview": True}
    if options_dict is not None:
        opts.update(options_dict)
    options = backend.parse_options(opts)
    module = _parse_ttir(tmp_path, backend, ttir)
    metadata = {}

    backend.make_ttgir_preview(module, metadata, options)

    return metadata["wave_ttgir_preview"]


def _accelerated_ttgir_text(tmp_path, ttir, options_dict=None):
    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    opts = {"num_warps": 1}
    if options_dict is not None:
        opts.update(options_dict)
    options = backend.parse_options(opts)
    module = _parse_ttir(tmp_path, backend, ttir)

    backend.make_ttgir(module, {}, options)

    return str(module)


def _accelerated_ttgir_convert_layout_types(tmp_path, ttir):
    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    module = _parse_ttir(tmp_path, backend, ttir)
    pairs = []

    backend.make_ttgir(module, {}, options)

    def collect(op):
        if op.get_name() == "ttg.convert_layout":
            pairs.append((str(op.get_operand(0).get_type()), str(op.get_result(0).get_type())))

    module.walk(collect)
    return pairs


@pytest.mark.parametrize(
    ("ttir", "kernel_name", "expected_min_blocked"),
    [
        (DOT_MATMUL_TTIR, "dot_kernel", 3),
        (DOT_MATMUL_32X32_TTIR, "dot_32x32_kernel", 3),
        (REALISTIC_MATMUL_TILE_TTIR, "realistic_matmul_tile_kernel", 1),
    ],
)
def test_wave_amd_ttgir_preview_accelerates_matmul_to_amd_wmma_encoding(tmp_path, ttir, kernel_name,
                                                                        expected_min_blocked):
    ttgir = _ttgir_preview_text(tmp_path, ttir)

    assert f"@{kernel_name}" in ttgir
    assert "ttg.target = \"hip:gfx1100\"" in ttgir
    assert "\"ttg.num-warps\" = 1" in ttgir
    assert "\"ttg.threads-per-warp\" = 32" in ttgir
    assert ttgir.count("#ttg.amd_wmma") == 1
    assert "#mma = #ttg.amd_wmma" in ttgir
    assert "version = 1" in ttgir
    assert "isTranspose = true" in ttgir
    assert "#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>" in ttgir
    assert "#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>" in ttgir
    assert ttgir.count("#ttg.blocked") >= expected_min_blocked
    assert ttgir.count("#ttg.dot_op") >= 4
    assert ttgir.count("ttg.convert_layout") >= 3
    assert ttgir.count("tt.load") == 2
    assert ttgir.count("tt.dot") == 1
    assert ttgir.count("tt.store") == 1


def test_wave_amd_ttgir_preview_preserves_matmul_k_loop_structure(tmp_path):
    ttgir = _ttgir_preview_text(tmp_path, REALISTIC_MATMUL_LOOP_TTIR)

    assert "@realistic_matmul_loop_kernel" in ttgir
    assert "scf.for" in ttgir
    assert ttgir.count("tt.load") == 2
    assert ttgir.count("tt.dot") == 1
    assert ttgir.count("tt.store") == 1
    assert "#ttg.amd_wmma" in ttgir
    assert "#ttg.dot_op" in ttgir
    assert "ttg.convert_layout" in ttgir


@pytest.mark.parametrize(
    ("ttir", "kernel_name", "expected_dot_ops"),
    [
        (DOT_MATMUL_K32_TTIR, "dot_k32_kernel", 1),
        (DOT_MATMUL_32X32_K32_TTIR, "dot_32x32_k32_kernel", 1),
        (REALISTIC_MATMUL_TILE_TTIR, "realistic_matmul_tile_kernel", 1),
        (REALISTIC_MATMUL_LOOP_TTIR, "realistic_matmul_loop_kernel", 1),
    ],
)
def test_wave_amd_ttgir_stage_accelerates_expanded_matmul_cases(tmp_path, ttir, kernel_name, expected_dot_ops):
    ttgir = _accelerated_ttgir_text(tmp_path, ttir)

    assert f"@{kernel_name}" in ttgir
    assert "#ttg.amd_wmma" in ttgir
    assert "#ttg.dot_op<{opIdx = 0" in ttgir
    assert "#ttg.dot_op<{opIdx = 1" in ttgir
    assert ttgir.count("tt.dot") == expected_dot_ops
    assert ttgir.count("ttg.convert_layout") >= 3


def test_wave_amd_ttgir_convert_layout_classifier_accepts_matrix_core_shapes():
    blocked = "tensor<16x16xf16, #ttg.blocked<{}>>"
    dot_a = ("tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_wmma<{version = 1}>, "
             "kWidth = 16}>>")
    dot_b = ("tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_wmma<{version = 1}>, "
             "kWidth = 16}>>")
    mma = "tensor<16x16xf32, #ttg.amd_wmma<{version = 1}>>"

    assert wave_lowering._ttgir_convert_layout_kind(blocked, dot_a) == "dot_operand"
    assert wave_lowering._ttgir_convert_layout_kind(blocked, dot_b) == "dot_operand"
    assert wave_lowering._ttgir_convert_layout_kind(blocked, mma) == "mma_accumulator"
    assert wave_lowering._ttgir_convert_layout_kind(mma, blocked) == "mma_result"
    assert wave_lowering._ttgir_convert_layout_kind(blocked, blocked) is None
    assert wave_lowering._ttgir_dot_operand_role(dot_a) == 0
    assert wave_lowering._ttgir_dot_operand_role(dot_b) == 1

    with pytest.raises(NotImplementedError, match="only matrix-core ttg.convert_layout ops"):
        wave_lowering._expect_ttgir_convert_layout_kind(blocked, blocked)


def test_wave_amd_dot_role_collection_uses_ttgir_encodings_only():

    class FakeValue:

        def __init__(self, value_id, value_type="tensor<16x16xf16, #ttg.blocked<{}>>"):
            self._id = value_id
            self._type = value_type

        def id(self):
            return self._id

        def get_type(self):
            return self._type

    class FakeOp:

        def __init__(self, name, operands, results=()):
            self._name = name
            self._operands = operands
            self._results = results

        def get_name(self):
            return self._name

        def get_operand(self, index):
            return self._operands[index]

        def get_result(self, index):
            return self._results[index]

    encoded_src = FakeValue(4)
    encoded_dst = FakeValue(5, "tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>")
    encoded_convert = FakeOp("ttg.convert_layout", (encoded_src, ), (encoded_dst, ))

    lowerer = wave_lowering._TTIRToWaveLowerer.__new__(wave_lowering._TTIRToWaveLowerer)
    lowerer.dot_operand_roles = {}
    lowerer.producers_by_result = {}
    lowerer._collect_dot_operand_roles((FakeOp("tt.dot", (FakeValue(1), FakeValue(2), FakeValue(3))), encoded_convert))

    assert lowerer.dot_operand_roles == {4: 1, 5: 1}


def test_wave_amd_ttgir_stage_uses_supported_matrix_core_convert_layouts(tmp_path):
    pairs = _accelerated_ttgir_convert_layout_types(tmp_path, DOT_MATMUL_TTIR)

    assert pairs
    assert [wave_lowering._ttgir_convert_layout_kind(src, dst) for src, dst in pairs] == [
        "dot_operand",
        "dot_operand",
        "mma_result",
    ]


def test_wave_amd_backend_hash_tracks_packaged_codegen_artifacts(tmp_path, monkeypatch):
    wave_translate = tmp_path / "wave-translate"
    pipelines = tmp_path / "pipelines.mlir"
    wave_translate.write_bytes(b"tool-v1")
    pipelines.write_text("pipeline-v1")
    monkeypatch.setattr(wave_compiler, "_wave_backend_artifact_paths", lambda: (wave_translate, pipelines))

    target = GPUTarget("wave_amd", "gfx1100", 32)
    first_hash = WaveAMDBackend(target).hash()
    wave_translate.write_bytes(b"tool-v2")

    assert WaveAMDBackend(target).hash() != first_hash


def test_wave_amd_blocked_layout_chunks_wide_1d_tensor():
    info = wave_lowering._TensorInfo((64, ), "i32")

    layout = wave_lowering._BlockedLayout.for_tensor(info, width=32, num_warps=1, num_ctas=1)

    assert layout.shape == (64, )
    assert layout.num_warps == 1
    assert layout.registers == 2


def test_wave_amd_blocked_layout_uses_warps_before_register_chunks():
    info = wave_lowering._TensorInfo((64, ), "i32")

    layout = wave_lowering._BlockedLayout.for_tensor(info, width=32, num_warps=2, num_ctas=1)

    assert layout.shape == (64, )
    assert layout.num_warps == 2
    assert layout.registers == 1


def test_wave_amd_blocked_layout_splits_wide_1d_tensor_across_ctas():
    info = wave_lowering._TensorInfo((64, ), "i32")

    layout = wave_lowering._BlockedLayout.for_tensor(info, width=32, num_warps=1, num_ctas=2)

    assert layout.shape == (64, )
    assert layout.num_warps == 1
    assert layout.num_ctas == 2
    assert layout.elements_per_cta == 32
    assert layout.registers == 1


def test_wave_amd_blocked_layout_chunks_2d_tensor():
    info = wave_lowering._TensorInfo((32, 32), "i32")

    layout = wave_lowering._BlockedLayout.for_tensor(info, width=32, num_warps=1, num_ctas=1)

    assert layout.shape == (32, 32)
    assert layout.registers == 32


def test_wave_amd_blocked_layout_chunks_2d_two_warps():
    info = wave_lowering._TensorInfo((32, 64), "i32")

    layout = wave_lowering._BlockedLayout.for_tensor(info, width=32, num_warps=2, num_ctas=1)

    assert layout.shape == (32, 64)
    assert layout.num_warps == 2
    assert layout.registers == 32


def test_wave_amd_blocked_layout_chunks_2d_two_ctas():
    info = wave_lowering._TensorInfo((32, 64), "i32")

    layout = wave_lowering._BlockedLayout.for_tensor(info, width=32, num_warps=1, num_ctas=2)

    assert layout.shape == (32, 64)
    assert layout.num_ctas == 2
    assert layout.elements_per_cta == 1024
    assert layout.registers == 32


def test_wave_amd_affine_index_tracks_dot_pointer_layouts():
    rows = wave_lowering._affine_coord(rank=2, axis=0)
    cols = wave_lowering._affine_coord(rank=2, axis=1)
    stride = wave_lowering._AffineIndex(16, (0, 0))

    row_major = wave_lowering._affine_add(wave_lowering._affine_mul(rows, stride), cols)
    col_major = wave_lowering._affine_add(wave_lowering._affine_mul(cols, stride), rows)

    assert row_major == wave_lowering._AffineIndex(0, (16, 1))
    assert col_major == wave_lowering._AffineIndex(0, (1, 16))


def test_wave_amd_make_wave_lowers_tiny_add(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, ELEMENTWISE_ADD_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "add_kernel"
    assert metadata["shared"] == 0
    assert 'waveamdmachine.target = "amdgcn-amd-amdhsa--gfx1100"' in wave
    assert "func.func @add_kernel" in wave
    assert "wave.kernel" in wave
    assert "wave.index_expr" in wave
    assert "lid" in wave
    assert "wave.load" in wave
    assert "wave.join" in wave
    assert "wave.fadd" in wave
    assert "wave.store" in wave
    assert "!wave.mem.token" in wave


def test_wave_amd_make_wave_lowers_wide_add_with_layout_chunks(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MULTI_REGISTER_ADD_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "wide_add_kernel"
    assert wave.count("wave.index_expr") == 2
    assert "lid + 32" in wave or "32 + lid" in wave
    assert wave.count("wave.load") == 4
    assert wave.count("wave.fadd") == 2
    assert wave.count("wave.store") == 2


def test_wave_amd_make_wave_lowers_wide_add_across_two_warps(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MULTI_REGISTER_ADD_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "wide_add_kernel"
    assert "wave.workitem_id 0" in wave
    assert "wi" in wave
    assert wave.count("wave.index_expr") == 1
    assert wave.count("wave.load") == 2
    assert wave.count("wave.fadd") == 1
    assert wave.count("wave.store") == 1


def test_wave_amd_make_wave_lowers_wide_add_across_two_ctas(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MULTI_REGISTER_ADD_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "wide_add_kernel"
    assert "wave.workgroup_id 0" in wave
    assert "wg_0" in wave
    assert "Mod" in wave or "mod" in wave
    assert wave.count("wave.load") == 2
    assert wave.count("wave.fadd") == 1
    assert wave.count("wave.store") == 1


def test_wave_amd_make_wave_lowers_program_id_across_two_ctas(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_program_id_axis(self, op):
            assert op.get_name() == "tt.get_program_id"
            return 0

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is not None:
                return value, "i32", None
            return 1.0, "f32", 64

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MULTI_CTA_PROGRAM_ID_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "store_multi_cta_pid_kernel"
    assert "wave.workgroup_id 0" in wave
    assert "Floor" in wave or "floor" in wave
    assert "Mod" in wave or "mod" in wave
    assert "wg_0" in wave
    assert wave.count("wave.store") == 1


def test_wave_amd_make_wave_lowers_2d_offsets_with_broadcasts(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is not None:
                return value, "i32", None
            return 1.0, "f32", 1024

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, TWO_D_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "store_2d_kernel"
    assert "wave.index_expr" in wave
    assert "floor" in wave or "Mod" in wave or "mod" in wave
    assert wave.count("wave.store") == 32


def test_wave_amd_make_wave_lowers_2d_offsets_across_two_warps(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is not None:
                return value, "i32", None
            return 1.0, "f32", 2048

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, TWO_D_MULTI_WARP_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "store_2d_multi_warp_kernel"
    assert "wave.workitem_id 0" in wave
    assert "wi" in wave
    assert "floor" in wave or "Mod" in wave or "mod" in wave
    assert wave.count("wave.store") == 32


def test_wave_amd_make_wave_lowers_2d_offsets_across_two_ctas(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is not None:
                return value, "i32", None
            return 1.0, "f32", 2048

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, TWO_D_MULTI_CTA_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "store_2d_multi_cta_kernel"
    assert "wave.workgroup_id 0" in wave
    assert "wg_0" in wave
    assert "Mod" in wave or "mod" in wave
    assert "floor" in wave or "Floor" in wave
    assert wave.count("wave.store") == 32


def _lower_accelerated_ttgir_to_wave_with_backend(tmp_path, ttir, options_dict=None):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    opts = {"num_warps": 1}
    if options_dict is not None:
        opts.update(options_dict)
    options = backend.parse_options(opts)
    metadata = {}
    module = _parse_ttir(tmp_path, backend, ttir)

    backend.make_ttgir(module, metadata, options)
    assert "#ttg.amd_wmma" in str(module)
    wave = backend.make_wave(module, metadata, options)
    return backend, options, wave, metadata


def _lower_accelerated_ttgir_to_wave(tmp_path, ttir, options_dict=None):
    _, _, wave, metadata = _lower_accelerated_ttgir_to_wave_with_backend(tmp_path, ttir, options_dict)
    return wave, metadata


def test_wave_amd_make_wave_lowers_accelerated_ttgir_dot_to_native_wmma(tmp_path):
    wave, metadata = _lower_accelerated_ttgir_to_wave(tmp_path, DOT_MATMUL_TTIR)

    assert metadata["name"] == "dot_kernel"
    assert "func.func @dot_kernel" in wave
    assert "waveamd.fragment_pack" in wave
    assert 'waveamd.mma "wmma.f32.16x16x16.f16"' in wave
    assert "waveamd.fragment_unpack" in wave
    assert "wave.store" in wave


def test_wave_amd_make_wave_lowers_accelerated_ttgir_dot_k32_to_two_native_wmma_ops(tmp_path):
    wave, metadata = _lower_accelerated_ttgir_to_wave(tmp_path, DOT_MATMUL_K32_TTIR)

    assert metadata["name"] == "dot_k32_kernel"
    assert "func.func @dot_k32_kernel" in wave
    assert wave.count('waveamd.mma "wmma.f32.16x16x16.f16"') == 2
    assert wave.count("waveamd.fragment_pack") == 4
    assert "wave.store" in wave


@pytest.mark.parametrize(
    ("ttir", "kernel_name", "expected_mmas", "expected_packs"),
    [
        (DOT_MATMUL_32X32_TTIR, "dot_32x32_kernel", 4, 4),
        (DOT_MATMUL_32X32_K32_TTIR, "dot_32x32_k32_kernel", 8, 8),
    ],
)
def test_wave_amd_make_wave_lowers_accelerated_ttgir_32x32_dot_to_native_wmma_grid(tmp_path, ttir, kernel_name,
                                                                                   expected_mmas, expected_packs):
    wave, metadata = _lower_accelerated_ttgir_to_wave(tmp_path, ttir)

    assert metadata["name"] == kernel_name
    assert f"func.func @{kernel_name}" in wave
    assert wave.count('waveamd.mma "wmma.f32.16x16x16.f16"') == expected_mmas
    assert wave.count("waveamd.fragment_pack") == expected_packs
    assert wave.count("waveamd.fragment_unpack") == 4
    assert wave.count("wave.store") == 32


@pytest.mark.parametrize(
    ("options_dict", "expected_marker"),
    [
        ({"num_warps": 2}, "wave.workitem_id 0"),
        ({"num_warps": 1, "num_ctas": 2}, "wave.workgroup_id 0"),
    ],
)
def test_wave_amd_make_wave_lowers_accelerated_ttgir_dot_scheduled_across_workers(tmp_path, options_dict,
                                                                                  expected_marker):
    wave, metadata = _lower_accelerated_ttgir_to_wave(tmp_path, DOT_MATMUL_32X32_TTIR, options_dict)

    assert metadata["name"] == "dot_32x32_kernel"
    assert expected_marker in wave
    assert wave.count("wave.where") >= 4
    assert wave.count('waveamd.mma "wmma.f32.16x16x16.f16"') == 4
    assert wave.count("wave.store") == 32


def test_wave_amd_make_wave_lowers_accelerated_ttgir_realistic_matmul_tile_pattern(tmp_path):
    wave, metadata = _lower_accelerated_ttgir_to_wave(tmp_path, REALISTIC_MATMUL_TILE_TTIR)

    assert metadata["name"] == "realistic_matmul_tile_kernel"
    assert "func.func @realistic_matmul_tile_kernel" in wave
    assert "wave.workgroup_id 0" in wave
    assert "wave.index_expr" in wave
    assert wave.count("wave.where") >= 3
    assert wave.count("waveamd.fragment_pack") == 2
    assert wave.count('waveamd.mma "wmma.f32.16x16x16.f16"') == 1
    assert "waveamd.fragment_unpack" in wave
    assert "wave.store" in wave


def test_wave_amd_make_wave_lowers_accelerated_ttgir_realistic_matmul_k_loop(tmp_path):
    wave, metadata = _lower_accelerated_ttgir_to_wave(tmp_path, REALISTIC_MATMUL_LOOP_TTIR)

    assert metadata["name"] == "realistic_matmul_loop_kernel"
    assert "scf.for" in wave
    assert wave.count('waveamd.mma "wmma.f32.16x16x16.f16"') == 1
    assert wave.count("waveamd.fragment_pack") == 2
    assert "waveamd.fragment_unpack" in wave
    assert "wave.store" in wave


def test_wave_amd_make_wave_lowers_same_mask_loads_and_store(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_program_id_axis(self, op):
            assert op.get_name() == "tt.get_program_id"
            return 0

        def get_cmpi_predicate(self, op):
            assert op.get_name() == "arith.cmpi"
            return "ult"

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is None:
                return None
            return value, "i32", None

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_ADD_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "masked_add_kernel"
    assert "wave.workgroup_id 0" in wave
    assert "wave.cmpi" in wave
    assert wave.count("wave.where") == 3
    assert "-> !wave.simd<f32, 32>, !wave.mem.token" in wave
    assert wave.count("wave.load") == 2
    assert "wave.fadd" in wave
    assert "wave.store" in wave
    assert "pid_0" in wave
    assert "lid" in wave


def test_wave_amd_make_wave_lowers_masked_load_other(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_program_id_axis(self, op):
            assert op.get_name() == "tt.get_program_id"
            return 0

        def get_cmpi_predicate(self, op):
            assert op.get_name() == "arith.cmpi"
            return "ult"

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is not None:
                return value, "i32", None
            return 5.0, "f32", 32

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_LOAD_OTHER_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "masked_load_other_kernel"
    assert wave.count("wave.where") == 2
    assert "otherwise" in wave
    assert "5.000000e+00" in wave
    assert wave.count("wave.load") == 1
    assert "wave.store" in wave


def test_wave_amd_make_wave_lowers_masked_sub_mul(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_program_id_axis(self, op):
            assert op.get_name() == "tt.get_program_id"
            return 0

        def get_cmpi_predicate(self, op):
            assert op.get_name() == "arith.cmpi"
            return "ult"

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is None:
                return None
            return value, "i32", None

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_SUB_MUL_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "masked_sub_mul_kernel"
    assert "wave.fsub" in wave
    assert "wave.fmul" in wave
    assert "arith.constant -1" in wave
    assert wave.count("wave.where") == 3
    assert wave.count("wave.load") == 2
    assert "wave.store" in wave


def test_wave_amd_make_wave_lowers_masked_select(tmp_path, monkeypatch):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )

    class FakeNative:

        def get_program_id_axis(self, op):
            assert op.get_name() == "tt.get_program_id"
            return 0

        def get_cmpi_predicate(self, op):
            assert op.get_name() == "arith.cmpi"
            return "ult"

        def get_arith_constant_splat(self, op):
            value = op.get_constant_value()
            if value is None:
                return None
            return value, "i32", None

        def get_result_tensor_info(self, op, result_index):
            return _fake_result_tensor_info(op, result_index)

    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: FakeNative())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_SELECT_TTIR)

    wave = backend.make_wave(module, metadata, options)

    assert metadata["name"] == "masked_select_kernel"
    assert "wave.select" in wave
    assert wave.count("wave.cmpi") == 2
    assert wave.count("wave.where") == 3
    assert wave.count("wave.load") == 2
    assert "wave.store" in wave


def test_wave_amd_make_amdgcn_emits_masked_kernel_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_ADD_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "masked_add_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "masked_add_kernel:" in amdgcn
    assert "global_load_b32" in amdgcn
    assert "global_store_b32" in amdgcn


def test_wave_amd_make_amdgcn_emits_masked_select_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_SELECT_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "masked_select_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "masked_select_kernel:" in amdgcn
    assert "v_cndmask_b32" in amdgcn
    assert "global_store_b32" in amdgcn


def test_wave_amd_make_amdgcn_emits_masked_load_other_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_LOAD_OTHER_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "masked_load_other_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "masked_load_other_kernel:" in amdgcn
    assert "global_load_b32" in amdgcn
    assert "global_store_b32" in amdgcn
    assert amdgcn.count("global_store_b32") >= 2


def test_wave_amd_make_amdgcn_emits_masked_sub_mul_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_SUB_MUL_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "masked_sub_mul_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "masked_sub_mul_kernel:" in amdgcn
    assert "v_sub_f32" in amdgcn
    assert "v_mul_f32" in amdgcn
    assert "global_store_b32" in amdgcn


def test_wave_amd_make_amdgcn_emits_2d_multi_warp_store_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, TWO_D_MULTI_WARP_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "store_2d_multi_warp_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "store_2d_multi_warp_kernel:" in amdgcn
    assert "global_store_b32" in amdgcn


def test_wave_amd_make_amdgcn_emits_2d_multi_cta_store_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, TWO_D_MULTI_CTA_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "store_2d_multi_cta_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "store_2d_multi_cta_kernel:" in amdgcn
    assert "global_store_b32" in amdgcn


def test_wave_amd_make_amdgcn_emits_multi_cta_program_id_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MULTI_CTA_PROGRAM_ID_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "store_multi_cta_pid_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "store_multi_cta_pid_kernel:" in amdgcn
    assert "global_store_b32" in amdgcn


def test_wave_amd_make_amdgcn_emits_dot_with_packaged_wave_translate(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    backend, options, wave, metadata = _lower_accelerated_ttgir_to_wave_with_backend(tmp_path, DOT_MATMUL_TTIR)
    amdgcn = backend.make_amdgcn(wave, metadata, options)

    assert metadata["name"] == "dot_kernel"
    assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"' in amdgcn
    assert "dot_kernel:" in amdgcn
    assert "global_load" in amdgcn
    assert "global_store" in amdgcn


def test_wave_amd_make_hsaco_emits_masked_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_ADD_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "masked_add_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def test_wave_amd_make_hsaco_emits_masked_select_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_SELECT_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "masked_select_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def test_wave_amd_make_hsaco_emits_masked_load_other_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_LOAD_OTHER_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "masked_load_other_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def test_wave_amd_make_hsaco_emits_masked_sub_mul_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MASKED_SUB_MUL_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "masked_sub_mul_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def test_wave_amd_make_hsaco_emits_2d_multi_warp_store_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, TWO_D_MULTI_WARP_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "store_2d_multi_warp_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def test_wave_amd_make_hsaco_emits_2d_multi_cta_store_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, TWO_D_MULTI_CTA_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "store_2d_multi_cta_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def test_wave_amd_make_hsaco_emits_multi_cta_program_id_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 2})
    metadata = {}
    module = _parse_ttir(tmp_path, backend, MULTI_CTA_PROGRAM_ID_STORE_TTIR)

    wave = backend.make_wave(module, metadata, options)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "store_multi_cta_pid_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def test_wave_amd_make_hsaco_emits_dot_kernel_elf(tmp_path):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    backend, options, wave, metadata = _lower_accelerated_ttgir_to_wave_with_backend(tmp_path, DOT_MATMUL_TTIR)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "dot_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


@pytest.mark.parametrize("options_dict", [
    {"num_warps": 2},
    {"num_warps": 1, "num_ctas": 2},
])
def test_wave_amd_make_hsaco_emits_scheduled_dot_kernel_elf(tmp_path, options_dict):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    wave_translate = wave_emission._packaged_wave_translate()
    if not wave_translate.is_file():
        pytest.skip("packaged wave-translate is required")

    backend, options, wave, metadata = _lower_accelerated_ttgir_to_wave_with_backend(
        tmp_path, DOT_MATMUL_32X32_TTIR, options_dict)
    amdgcn = backend.make_amdgcn(wave, metadata, options)
    hsaco = backend.make_hsaco(amdgcn, metadata, options)

    assert metadata["name"] == "dot_32x32_kernel"
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert len(hsaco) > 0


def _require_wave_amd_runtime(tmp_path, monkeypatch, device):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")

    from triton.backends import Backend, backends

    monkeypatch.setitem(backends, "wave_amd", Backend(WaveAMDBackend, wave_driver.WaveAMDDriver))
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    return torch, target


@pytest.mark.parametrize(
    ("m", "n", "k", "kernel_name"),
    [
        (16, 16, 16, "dot_e2e_kernel"),
        (16, 16, 32, "dot_k32_e2e_kernel"),
        (32, 32, 16, "dot_32x32_e2e_kernel"),
    ],
)
def test_wave_amd_runtime_launches_dot_matmul_tile(tmp_path, monkeypatch, device, m, n, k, kernel_name):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")

    from triton.backends import Backend, backends

    monkeypatch.setitem(backends, "wave_amd", Backend(WaveAMDBackend, wave_driver.WaveAMDDriver))
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))

    a_matrix = torch.arange(m * k, device=device, dtype=torch.float32).reshape(m, k) / 32.0
    b_matrix = (torch.arange(k * n, device=device, dtype=torch.float32).reshape(k, n) / 64.0) - 1.0
    a = a_matrix.to(torch.float16).contiguous()
    b = b_matrix.to(torch.float16).t().contiguous()
    c = torch.full((m, n), -999.0, device=device, dtype=torch.float32)

    module_path = tmp_path / f"dot_matmul_{m}x{n}x{k}.ttir"
    module_path.write_text(_dot_matmul_ttir(m, n, k, kernel_name))
    kernel = triton_compiler.compile(
        str(module_path),
        target=GPUTarget("wave_amd", target.arch, target.warp_size),
        options={"num_warps": 1},
    )
    kernel[(1, 1, 1)](a, b, c)
    getattr(torch, device).synchronize()

    expected = a.to(torch.float32) @ b_matrix.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)


def test_wave_amd_runtime_launches_looped_dot_matmul_tile(tmp_path, monkeypatch, device):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")

    from triton.backends import Backend, backends

    monkeypatch.setitem(backends, "wave_amd", Backend(WaveAMDBackend, wave_driver.WaveAMDDriver))
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))

    m = n = 16
    k = 32
    a_matrix = torch.arange(m * k, device=device, dtype=torch.float32).reshape(m, k) / 32.0
    b_matrix = (torch.arange(k * n, device=device, dtype=torch.float32).reshape(k, n) / 64.0) - 1.0
    a = a_matrix.to(torch.float16).contiguous()
    b = b_matrix.to(torch.float16).t().contiguous()
    c = torch.full((m, n), -999.0, device=device, dtype=torch.float32)

    module_path = tmp_path / "looped_dot_matmul.ttir"
    module_path.write_text(REALISTIC_MATMUL_LOOP_TTIR)
    kernel = triton_compiler.compile(
        str(module_path),
        target=GPUTarget("wave_amd", target.arch, target.warp_size),
        options={"num_warps": 1},
    )
    kernel[(1, 1, 1)](a, b, c)
    getattr(torch, device).synchronize()

    expected = a.to(torch.float32) @ b_matrix.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(("m", "n", "k", "num_warps"), [
    (16, 16, 16, 1),
    (16, 16, 32, 1),
    (32, 32, 16, 4),
    (32, 32, 32, 4),
])
def test_wave_amd_e2e_triton_jit_matmul_tile(tmp_path, monkeypatch, device, m, n, k, num_warps):
    torch, _ = _require_wave_amd_runtime(tmp_path, monkeypatch, device)

    @triton.jit
    def matmul_tile_kernel(a, b, c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a + offs_m[:, None] * BLOCK_K + offs_k[None, :]
        b_ptrs = b + offs_k[:, None] + offs_n[None, :] * BLOCK_K
        a_tile = tl.load(a_ptrs)
        b_tile = tl.load(b_ptrs)
        acc = tl.dot(a_tile, b_tile)
        c_ptrs = c + offs_m[:, None] * BLOCK_N + offs_n[None, :]
        tl.store(c_ptrs, acc)

    a_matrix = torch.arange(m * k, device=device, dtype=torch.float32).reshape(m, k) / 32.0
    b_matrix = (torch.arange(k * n, device=device, dtype=torch.float32).reshape(k, n) / 64.0) - 1.0
    a = a_matrix.to(torch.float16).contiguous()
    b = b_matrix.to(torch.float16).t().contiguous()
    c = torch.full((m, n), -999.0, device=device, dtype=torch.float32)

    matmul_tile_kernel[(1, 1, 1)](a, b, c, BLOCK_M=m, BLOCK_N=n, BLOCK_K=k, num_warps=num_warps)
    getattr(torch, device).synchronize()

    expected = a.to(torch.float32) @ b_matrix.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)


def test_wave_amd_e2e_triton_jit_looped_matmul(tmp_path, monkeypatch, device):
    torch, _ = _require_wave_amd_runtime(tmp_path, monkeypatch, device)

    @triton.jit
    def looped_matmul_kernel(a, b, c, k_tiles):
        offs_m = tl.arange(0, 16)
        offs_n = tl.arange(0, 16)
        offs_k = tl.arange(0, 16)
        acc = tl.zeros((16, 16), dtype=tl.float32)
        for k_tile in range(0, k_tiles):
            k_base = k_tile * 16
            a_ptrs = a + offs_m[:, None] * 32 + (k_base + offs_k)[None, :]
            b_ptrs = b + (k_base + offs_k)[:, None] + offs_n[None, :] * 32
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            acc += tl.dot(a_tile, b_tile)
        c_ptrs = c + offs_m[:, None] * 16 + offs_n[None, :]
        tl.store(c_ptrs, acc)

    m = n = 16
    k = 32
    a_matrix = torch.arange(m * k, device=device, dtype=torch.float32).reshape(m, k) / 32.0
    b_matrix = (torch.arange(k * n, device=device, dtype=torch.float32).reshape(k, n) / 64.0) - 1.0
    a = a_matrix.to(torch.float16).contiguous()
    b = b_matrix.to(torch.float16).t().contiguous()
    c = torch.full((m, n), -999.0, device=device, dtype=torch.float32)

    looped_matmul_kernel[(1, 1, 1)](a, b, c, 2, num_warps=1)
    getattr(torch, device).synchronize()

    expected = a.to(torch.float32) @ b_matrix.to(torch.float16).to(torch.float32)
    torch.testing.assert_close(c, expected, rtol=1e-2, atol=1e-2)


def test_wave_amd_runtime_launches_realistic_matmul_tile_with_boundary_masks(tmp_path, monkeypatch, device):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")

    from triton.backends import Backend, backends

    monkeypatch.setitem(backends, "wave_amd", Backend(WaveAMDBackend, wave_driver.WaveAMDDriver))
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))

    m, n, k = 13, 11, 9
    a_ld = b_ld = c_ld = 16
    a_full = torch.full((16, a_ld), 3.0, device=device, dtype=torch.float16)
    b_matrix_full = torch.full((b_ld, 16), -2.0, device=device, dtype=torch.float16)
    a_values = (torch.arange(m * k, device=device, dtype=torch.float32).reshape(m, k) / 32.0).to(torch.float16)
    b_values = ((torch.arange(k * n, device=device, dtype=torch.float32).reshape(k, n) / 64.0) - 1.0).to(torch.float16)
    a_full[:m, :k] = a_values
    b_matrix_full[:k, :n] = b_values
    b_physical = b_matrix_full.t().contiguous()
    c = torch.full((16, c_ld), -999.0, device=device, dtype=torch.float32)

    module_path = tmp_path / "realistic_matmul_tile.ttir"
    module_path.write_text(BOUNDARY_MATMUL_TILE_TTIR)
    kernel = triton_compiler.compile(
        str(module_path),
        target=GPUTarget("wave_amd", target.arch, target.warp_size),
        options={"num_warps": 1},
    )
    kernel[(1, 1, 1)](a_full, b_physical, c, m, n, k, a_ld, 1, 1, b_ld, c_ld, 1)
    getattr(torch, device).synchronize()

    expected = a_values.to(torch.float32) @ b_values.to(torch.float32)
    torch.testing.assert_close(c[:m, :n], expected, rtol=1e-2, atol=1e-2)
    assert torch.all(c[m:, :] == -999.0)
    assert torch.all(c[:, n:] == -999.0)


def test_wave_amd_runtime_launches_masked_kernel_tail(tmp_path, monkeypatch, device):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)

    n = 45
    total = 64
    a = torch.arange(total, device=device, dtype=torch.float32)
    b = torch.arange(total, device=device, dtype=torch.float32) * 2.0
    c = torch.full((total, ), -7.0, device=device, dtype=torch.float32)
    module_path = tmp_path / "masked_add.ttir"
    module_path.write_text(MASKED_ADD_TTIR)
    kernel = triton_compiler.compile(
        str(module_path),
        target=GPUTarget("wave_amd", target.arch, target.warp_size),
        options={"num_warps": 1},
    )
    kernel[(2, 1, 1)](a, b, c, n)
    getattr(torch, device).synchronize()

    expected = torch.full((total, ), -7.0, device=device, dtype=torch.float32)
    expected[:n] = a[:n] + b[:n]
    torch.testing.assert_close(c, expected)


def test_wave_amd_runtime_launches_masked_sub_mul_tail(tmp_path, monkeypatch, device):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)

    n = 45
    total = 64
    a = torch.arange(total, device=device, dtype=torch.float32)
    b = torch.arange(total, device=device, dtype=torch.float32) * 0.5
    c = torch.full((total, ), -7.0, device=device, dtype=torch.float32)
    module_path = tmp_path / "masked_sub_mul.ttir"
    module_path.write_text(MASKED_SUB_MUL_TTIR)
    kernel = triton_compiler.compile(
        str(module_path),
        target=GPUTarget("wave_amd", target.arch, target.warp_size),
        options={"num_warps": 1},
    )
    kernel[(2, 1, 1)](a, b, c, n)
    getattr(torch, device).synchronize()

    expected = torch.full((total, ), -7.0, device=device, dtype=torch.float32)
    expected[:n] = (a[:n] - b[:n]) * b[:n]
    torch.testing.assert_close(c, expected)


def test_wave_amd_runtime_launches_masked_select_tail(tmp_path, monkeypatch, device):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)

    n = 45
    total = 64
    a = torch.arange(total, device=device, dtype=torch.float32)
    b = torch.arange(total, device=device, dtype=torch.float32) * 10.0
    c = torch.full((total, ), -7.0, device=device, dtype=torch.float32)
    module_path = tmp_path / "masked_select.ttir"
    module_path.write_text(MASKED_SELECT_TTIR)
    kernel = triton_compiler.compile(
        str(module_path),
        target=GPUTarget("wave_amd", target.arch, target.warp_size),
        options={"num_warps": 1},
    )
    kernel[(2, 1, 1)](a, b, c, n)
    getattr(torch, device).synchronize()

    expected = torch.full((total, ), -7.0, device=device, dtype=torch.float32)
    expected[:16] = a[:16]
    expected[16:32] = b[16:32]
    expected[32:n] = a[32:n]
    torch.testing.assert_close(c, expected)


def test_wave_amd_runtime_launches_masked_load_other_tail(tmp_path, monkeypatch, device):
    pytest.importorskip(
        "mlir.dialects.wave_dsl",
        reason="Wave Python MLIR builder bindings are required",
    )
    pytest.importorskip("triton._C.libtriton.wave_amd")
    pytest.importorskip("triton._C.libtriton.amd")
    if device != "cuda":
        pytest.skip("Wave AMD runtime smoke requires a CUDA/HIP torch device")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("Wave AMD runtime smoke requires ROCm PyTorch and an active HIP device")

    try:
        active_driver = wave_driver.WaveAMDDriver()
        target = active_driver.get_current_target()
    except Exception as exc:
        pytest.skip(f"Wave AMD runtime smoke requires a working HIP runtime: {exc}")
    monkeypatch.setattr(triton_compiler.driver, "_default", active_driver)
    monkeypatch.setattr(triton_compiler.driver, "_active", active_driver)

    n = 45
    total = 64
    a = torch.arange(total, device=device, dtype=torch.float32)
    c = torch.full((total, ), -7.0, device=device, dtype=torch.float32)
    module_path = tmp_path / "masked_load_other.ttir"
    module_path.write_text(MASKED_LOAD_OTHER_TTIR)
    kernel = triton_compiler.compile(
        str(module_path),
        target=GPUTarget("wave_amd", target.arch, target.warp_size),
        options={"num_warps": 1},
    )
    kernel[(2, 1, 1)](a, c, n)
    getattr(torch, device).synchronize()

    expected = torch.full((total, ), 5.0, device=device, dtype=torch.float32)
    expected[:n] = a[:n]
    torch.testing.assert_close(c, expected)


def test_wave_amd_make_wave_rejects_textual_ttir():
    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({})

    with pytest.raises(TypeError, match="module object"):
        backend.make_wave(ELEMENTWISE_ADD_TTIR, {}, options)


def test_wave_amd_lowering_reads_structural_attrs_through_native_helpers(monkeypatch):

    class FakeNative:

        def __init__(self):
            self.calls = []

        def get_program_id_axis(self, op):
            self.calls.append(("program_id", op))
            return 1

        def get_cmpi_predicate(self, op):
            self.calls.append(("cmpi", op))
            return "ult"

        def get_arith_constant_splat(self, op):
            self.calls.append(("constant", op))
            return 7, "i32", None

        def get_result_tensor_info(self, op, result_index):
            self.calls.append(("tensor_info", op, result_index))
            return None

    fake_native = FakeNative()
    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: fake_native)
    lowerer = wave_lowering._TTIRToWaveLowerer.__new__(wave_lowering._TTIRToWaveLowerer)
    program_id_op = object()
    cmpi_op = object()

    assert lowerer._program_id_axis(program_id_op) == 1
    assert lowerer._cmpi_predicate(cmpi_op) == "ult"
    assert lowerer._arith_constant_splat(cmpi_op) == (7, "i32", None)
    assert fake_native.calls == [("program_id", program_id_op), ("cmpi", cmpi_op), ("constant", cmpi_op)]


def test_wave_amd_make_amdgcn_uses_packaged_wave_translate(tmp_path, monkeypatch):
    fake_wave_translate = tmp_path / "wave-translate"
    fake_wave_translate.write_text("#!/bin/sh\nprintf 'fake amdgcn\\n'\n")
    fake_wave_translate.chmod(0o755)
    monkeypatch.setattr(wave_emission, "_packaged_wave_translate", lambda: fake_wave_translate)

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({})
    metadata = {
        "name": "add_kernel",
        "shared": 0,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
    }

    wave_mlir = ('module attributes {waveamdmachine.target = "amdgcn-amd-amdhsa--gfx1100"} {}')
    amdgcn = backend.make_amdgcn(wave_mlir, metadata, options)

    assert amdgcn == "fake amdgcn\n"
    assert metadata == {
        "name": "add_kernel",
        "shared": 0,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
    }


def test_wave_amd_make_amdgcn_missing_tool_has_clear_diagnostic(tmp_path, monkeypatch):
    missing_tool = tmp_path / "missing-wave-translate"
    monkeypatch.setattr(wave_emission, "_packaged_wave_translate", lambda: missing_tool)

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({})

    with pytest.raises(RuntimeError) as excinfo:
        backend.make_amdgcn("module {}", {"name": "add_kernel"}, options)

    message = str(excinfo.value)
    assert "wave-translate" in message
    assert str(missing_tool) in message
    assert "Rebuild Triton" in message


def test_wave_amd_make_hsaco_uses_triton_amd_codegen_helpers(monkeypatch):

    class FakeAMDCodegen:

        def __init__(self):
            self.assembled = None
            self.linked = False

        def assemble_amdgcn(self, amdgcn, arch, features):
            self.assembled = (amdgcn, arch, features)
            return b"fake object"

        def link_hsaco(self, in_path, out_path):
            assert Path(in_path).read_bytes() == b"fake object"
            Path(out_path).write_bytes(b"fake hsaco")
            self.linked = True

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({})
    metadata = {
        "name": "add_kernel",
        "shared": 0,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
    }
    fake_amd = FakeAMDCodegen()
    monkeypatch.setattr(wave_emission, "_triton_amd_codegen", lambda: fake_amd)

    hsaco = backend.make_hsaco("fake amdgcn", metadata, options)

    assert isinstance(hsaco, bytes)
    assert hsaco == b"fake hsaco"
    assert fake_amd.assembled == ("fake amdgcn", "gfx1100", "")
    assert fake_amd.linked
    assert metadata == {
        "name": "add_kernel",
        "shared": 0,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
    }


def test_wave_amd_make_hsaco_link_failure_has_clear_diagnostic(monkeypatch):

    class FakeAMDCodegen:

        def assemble_amdgcn(self, amdgcn, arch, features):
            return b"fake object"

        def link_hsaco(self, in_path, out_path):
            raise RuntimeError("fake lld failure")

    monkeypatch.setattr(wave_emission, "_triton_amd_codegen", lambda: FakeAMDCodegen())

    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({})

    with pytest.raises(RuntimeError) as excinfo:
        backend.make_hsaco("fake amdgcn", {"name": "add_kernel"}, options)

    message = str(excinfo.value)
    assert "HSACO emission failed while linking" in message
    assert "gfx1100" in message
    assert "fake lld failure" in message


def test_wave_amd_compiled_kernel_loads_hsaco_with_hip_runtime_contract(tmp_path, monkeypatch):
    target = GPUTarget("wave_amd", "gfx1100", 32)
    backend = WaveAMDBackend(target)
    metadata = {
        "hash": "hash",
        "target": {
            "backend": target.backend,
            "arch": target.arch,
            "warp_size": target.warp_size,
        },
        "num_warps": 1,
        "num_ctas": 1,
        "shared": 0,
        "warp_size": 32,
        "launch_cooperative_grid": False,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "profile_scratch_size": 0,
        "profile_scratch_align": 1,
        "tensordesc_meta": {},
        "name": "add_kernel",
    }
    metadata_path = tmp_path / "kernel.json"
    hsaco_path = tmp_path / "kernel.hsaco"
    metadata_path.write_text(json.dumps(metadata))
    hsaco_path.write_bytes(b"fake hsaco")

    load_calls = []

    class FakeUtils:

        def load_binary(self, name, kernel, shared, device):
            load_calls.append((name, kernel, shared, device))
            return "module", "function", 0, 0, 1024

        def unload_module(self, module):
            pass

    class FakeLauncher:

        def __init__(self, src, metadata):
            self.src = src
            self.metadata = metadata

        def __call__(self, *args, **kwargs):
            pass

    class FakeDriver:
        utils = FakeUtils()
        launcher_cls = FakeLauncher

        def get_current_device(self):
            return 7

        def get_current_target(self):
            return target

    monkeypatch.setattr(triton_compiler, "make_backend", lambda loaded_target: backend)
    monkeypatch.setattr(triton_compiler, "max_shared_mem", lambda device: 1024)
    monkeypatch.setattr(triton_compiler.driver, "_active", FakeDriver())

    src = SimpleNamespace(signature={}, constants={})
    kernel = triton_compiler.CompiledKernel(
        src,
        {
            "kernel.json": str(metadata_path),
            "kernel.hsaco": str(hsaco_path),
        },
        "hash",
    )
    kernel._init_handles()

    assert kernel.kernel == b"fake hsaco"
    assert load_calls == [("add_kernel", b"fake hsaco", 0, 7)]
    assert kernel.module == "module"
    assert kernel.function == "function"


def test_wave_amd_driver_exports_one_concrete_driver():
    drivers = [
        getattr(wave_driver, name) for name in dir(wave_driver) if inspect.isclass(getattr(wave_driver, name))
        and issubclass(getattr(wave_driver, name), DriverBase) and not inspect.isabstract(getattr(wave_driver, name))
    ]

    assert drivers == [wave_driver.WaveAMDDriver]


def test_wave_amd_driver_activation_requires_explicit_selection(monkeypatch):
    monkeypatch.setattr(hip_driver.HIPDriver, "is_active", staticmethod(lambda: True))
    monkeypatch.delenv("TRITON_WAVE_AMD_ENABLE", raising=False)
    monkeypatch.delenv("TRITON_DEFAULT_BACKEND", raising=False)

    assert not wave_driver.WaveAMDDriver.is_active()

    monkeypatch.setenv("TRITON_WAVE_AMD_ENABLE", "1")
    assert not wave_driver.WaveAMDDriver.is_active()

    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "wave_amd")
    assert wave_driver.WaveAMDDriver.is_active()

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

libtriton = pytest.importorskip("triton._C.libtriton")
compiler_api = pytest.importorskip("triton.backends.compiler")
triton_compiler = pytest.importorskip("triton.compiler.compiler")
driver_api = pytest.importorskip("triton.backends.driver")
wave_compiler = pytest.importorskip("triton.backends.wave_amd.compiler")
wave_emission = pytest.importorskip("triton.backends.wave_amd.emission")
wave_lowering = pytest.importorskip("triton.backends.wave_amd.lowering")
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
    assert list(stages) == ["ttir", "wave", "amdgcn", "hsaco"]


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
    assert layout.registers == 2


def test_wave_amd_blocked_layout_chunks_2d_tensor():
    info = wave_lowering._TensorInfo((32, 32), "i32")

    layout = wave_lowering._BlockedLayout.for_tensor(info, width=32, num_warps=1, num_ctas=1)

    assert layout.shape == (32, 32)
    assert layout.registers == 32


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

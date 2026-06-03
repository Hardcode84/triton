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


def _parse_ttir(tmp_path, backend, ttir):
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    path = tmp_path / "kernel.ttir"
    path.write_text(ttir)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


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
    assert "gpu.container_module" in wave
    assert "gpu.kernel" in wave
    assert "wave.kernel" in wave
    assert "wave.index_expr" in wave
    assert "lid" in wave
    assert "wave.load" in wave
    assert "wave.join" in wave
    assert "wave.fadd" in wave
    assert "wave.store" in wave
    assert "!wave.mem.token" in wave


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

    fake_native = FakeNative()
    monkeypatch.setattr(wave_lowering, "_wave_amd_native", lambda: fake_native)
    lowerer = wave_lowering._TTIRToWaveLowerer.__new__(wave_lowering._TTIRToWaveLowerer)
    program_id_op = object()
    cmpi_op = object()

    assert lowerer._program_id_axis(program_id_op) == 1
    assert lowerer._cmpi_predicate(cmpi_op) == "ult"
    assert fake_native.calls == [("program_id", program_id_op), ("cmpi", cmpi_op)]


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
    monkeypatch.setattr(triton_compiler.driver, "active", FakeDriver())

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

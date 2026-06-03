import inspect
from types import SimpleNamespace

import pytest

compiler_api = pytest.importorskip("triton.backends.compiler")
driver_api = pytest.importorskip("triton.backends.driver")
wave_compiler = pytest.importorskip("triton.backends.wave_amd.compiler")
wave_driver = pytest.importorskip("triton.backends.wave_amd.driver")
hip_driver = pytest.importorskip("triton.backends.amd.driver")

GPUTarget = compiler_api.GPUTarget
Language = compiler_api.Language
DriverBase = driver_api.DriverBase
WaveAMDBackend = wave_compiler.WaveAMDBackend


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
    assert list(stages) == ["ttir", "wave"]
    with pytest.raises(NotImplementedError, match="TTIR-to-Wave lowering is not implemented yet"):
        stages["wave"](object(), {})


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

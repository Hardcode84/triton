import os

from triton.backends.compiler import GPUTarget
from triton.backends.amd import driver as amd_driver
from triton import knobs

ENABLE_ENV = "TRITON_WAVE_AMD_ENABLE"


class WaveAMDDriver(amd_driver.HIPDriver):
    """HIP-backed driver for the ``wave_amd`` backend.

    This driver is identical to Triton's upstream ``HIPDriver`` (same
    ``HIPUtils``, ``HIPLauncher``, device interface, etc.); it only changes the
    reported target backend name to ``wave_amd`` and is gated off by default.
    It becomes active only when ``wave_amd`` is explicitly selected, so normal
    HIP usage is completely unaffected unless a user opts in.
    """

    @staticmethod
    def is_active():
        if os.environ.get(ENABLE_ENV) != "1":
            return False
        if os.environ.get("TRITON_DEFAULT_BACKEND") != "wave_amd":
            return False
        return amd_driver.HIPDriver.is_active()

    def get_current_target(self):
        device = self.get_current_device()
        props = self.utils.get_device_properties(device)
        arch = knobs.runtime.override_arch or props["arch"]
        warp_size = props["warpSize"]
        return GPUTarget("wave_amd", arch.split(":")[0], warp_size)

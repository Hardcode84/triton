import os

from triton import knobs
from triton.backends.amd import driver as hip_driver
from triton.backends.compiler import GPUTarget

ENABLE_ENV = "TRITON_WAVE_AMD_ENABLE"


class WaveAMDDriver(hip_driver.HIPDriver):

    @staticmethod
    def is_active():
        if os.environ.get(ENABLE_ENV) != "1":
            return False
        if os.environ.get("TRITON_DEFAULT_BACKEND") != "wave_amd":
            return False
        return hip_driver.HIPDriver.is_active()

    def get_current_target(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = knobs.runtime.override_arch or device_properties["arch"]
        warp_size = device_properties["warpSize"]
        return GPUTarget("wave_amd", arch.split(":")[0], warp_size)

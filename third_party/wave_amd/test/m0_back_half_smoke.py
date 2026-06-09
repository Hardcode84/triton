#!/usr/bin/env python3
"""M0 back-half smoke test for the wave_amd backend (no bridge).

Validates the wave_amd back-half toolchain end to end, independent of the
TTGIR -> Wave converter (which lands in later milestones):

    high-level Wave MLIR
      -> wave-translate --wave-to-amdgpu-asm     (wave's LLVM: AMDGPU asm)
      -> amd.assemble_amdgcn + amd.link_hsaco    (triton's LLVM: HSACO)
      -> HIP load + launch on gfx1100            (numeric check)

This exercises emit + assemble + link + launch + the asm->HSACO cross-toolchain
code-object contract + the kernarg ABI, with both statically-linked LLVM
runtimes in play (wave's via the wave-translate subprocess, triton's via _C).

The fixture below is the high-level Wave IR emitted by
``wavec --emit=wave --offload-arch=gfx1100 -DW=32`` for the saxpy kernel
(test/Integration/Inputs/wavec_saxpy_runtime.wave in the wave-mlir repo). saxpy
is a masked elementwise kernel (``where (i < n) { y = a*x + y }``), i.e. the same
op class as the eventual M3 masked-copy target.

Usage:
    python third_party/wave_amd/test/m0_back_half_smoke.py

Environment overrides:
    WAVE_REPO       wave-mlir repo root        (default: /home/vano/7/7)
    WAVE_TRANSLATE  path to wave-translate     (default: $WAVE_REPO/build/bin/wave-translate)
    HIP_LIB         path to libamdhip64.so     (default: /opt/rocm-6.3.3/lib/libamdhip64.so)
    WAVE_AMD_CHIP   target chip                (default: gfx1100)
"""

from __future__ import annotations

import ctypes
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = "saxpy"
WAVE_SIZE = 32
WORKGROUP_SIZE = 32
ALPHA = 1.5

# High-level Wave IR (wavec --emit=wave, gfx1100, W=32) for saxpy.
SAXPY_WAVE_MLIR = """\
module attributes {gpu.container_module, waveamdmachine.target = "amdgcn-amd-amdhsa--gfx1100"} {
  func.func @saxpy(%arg0: !wave.ptr<#wave.global, f32>, %arg1: !wave.ptr<#wave.global, f32>, %arg2: f32, %arg3: i32) attributes {wave.kernel} {
    %0 = wave.lane_id : !wave.simd<i32, 32>
    %1 = wave.workgroup_id 0
    %c32_i32 = arith.constant 32 : i32
    %2 = wave.binary muli %1, %c32_i32 : i32, i32 -> i32
    %3 = wave.binary addi %2, %0 : i32, !wave.simd<i32, 32> -> !wave.simd<i32, 32>
    %4 = wave.splat %arg3 : i32 -> !wave.simd<i32, 32>
    %5 = wave.cmpi ult %3, %4 : !wave.simd<i32, 32>, !wave.simd<i32, 32> -> !wave.mask<32>
    wave.where %5 {
      %6 = wave.ptr_add %arg0, %3 : !wave.ptr<#wave.global, f32>, !wave.simd<i32, 32> -> !wave.simd<!wave.ptr<#wave.global, f32>, 32>
      %value, %token = wave.load %6 : (!wave.simd<!wave.ptr<#wave.global, f32>, 32>) -> (!wave.simd<f32, 32>, !wave.mem.token)
      %7 = wave.ptr_add %arg1, %3 : !wave.ptr<#wave.global, f32>, !wave.simd<i32, 32> -> !wave.simd<!wave.ptr<#wave.global, f32>, 32>
      %value_0, %token_1 = wave.load %7 : (!wave.simd<!wave.ptr<#wave.global, f32>, 32>) -> (!wave.simd<f32, 32>, !wave.mem.token)
      %8 = wave.splat %arg2 : f32 -> !wave.simd<f32, 32>
      %9 = wave.fmul %8, %value : !wave.simd<f32, 32>, !wave.simd<f32, 32> -> !wave.simd<f32, 32>
      %10 = wave.fadd %9, %value_0 : !wave.simd<f32, 32>, !wave.simd<f32, 32> -> !wave.simd<f32, 32>
      %11 = wave.ptr_add %arg1, %3 : !wave.ptr<#wave.global, f32>, !wave.simd<i32, 32> -> !wave.simd<!wave.ptr<#wave.global, f32>, 32>
      %12 = wave.store %10 -> %11 : (!wave.simd<f32, 32>, !wave.simd<!wave.ptr<#wave.global, f32>, 32>) -> !wave.mem.token
    } : !wave.mask<32>
    return
  }
}
"""

HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


def chip() -> str:
    return os.environ.get("WAVE_AMD_CHIP", "gfx1100")


def wave_repo() -> Path:
    return Path(os.environ.get("WAVE_REPO", "/home/vano/7/7"))


def wave_translate_bin() -> Path:
    env = os.environ.get("WAVE_TRANSLATE")
    if env:
        return Path(env)
    return wave_repo() / "build" / "bin" / "wave-translate"


def hip_lib() -> str:
    return os.environ.get("HIP_LIB", "/opt/rocm-6.3.3/lib/libamdhip64.so")


def emit_asm(module_text: str) -> str:
    """high-level Wave MLIR -> AMDGPU asm via wave-translate (wave's LLVM)."""
    tool = wave_translate_bin()
    if not tool.exists():
        raise SystemExit(f"wave-translate not found at {tool}; set WAVE_TRANSLATE")
    proc = subprocess.run(
        [str(tool), "--wave-to-amdgpu-asm", "-"],
        input=module_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"wave-translate failed ({proc.returncode})")
    return proc.stdout


def assemble_link(asm: str) -> bytes:
    """AMDGPU asm -> HSACO via triton's assembler/linker (triton's LLVM)."""
    from triton._C.libtriton import amd, llvm

    llvm.init_targets()
    # f32-only smoke; gfx11 f16/bf16 kernels would also need '-real-true16' in
    # features here (cf. triton's disable_real_true16_feature in third_party/amd).
    obj = amd.assemble_amdgcn(asm, chip(), "")
    with tempfile.NamedTemporaryFile() as tmp_out, tempfile.NamedTemporaryFile() as tmp_in:
        Path(tmp_in.name).write_bytes(obj)
        amd.link_hsaco(tmp_in.name, tmp_out.name)
        return Path(tmp_out.name).read_bytes()


class Hip:
    """Minimal HIP module/launch wrapper (kernelParams array-of-pointers ABI)."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        c_vp, c_vpp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        self._sig("hipInit", [ctypes.c_uint])
        self._sig("hipGetErrorString", [ctypes.c_int], ctypes.c_char_p)
        self._sig("hipMalloc", [c_vpp, ctypes.c_size_t])
        self._sig("hipFree", [c_vp])
        self._sig("hipMemcpy", [c_vp, c_vp, ctypes.c_size_t, ctypes.c_int])
        self._sig("hipModuleLoadData", [c_vpp, c_vp])
        self._sig("hipModuleUnload", [c_vp])
        self._sig("hipModuleGetFunction", [c_vpp, c_vp, ctypes.c_char_p])
        self._sig("hipModuleLaunchKernel", [c_vp] + [ctypes.c_uint] * 7 + [c_vp, c_vpp, c_vp])
        self._sig("hipDeviceSynchronize", [])

    def _sig(self, name, argtypes, restype=ctypes.c_int):
        fn = getattr(self.lib, name)
        fn.argtypes = argtypes
        fn.restype = restype

    def check(self, code: int, what: str):
        if code != 0:
            raw = self.lib.hipGetErrorString(code)
            raise RuntimeError(f"{what}: {raw.decode() if raw else code}")


def _ptr_to(value) -> ctypes.c_void_p:
    return ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)


def launch_and_check(hsaco: bytes) -> None:
    hip = Hip(hip_lib())
    hip.check(hip.lib.hipInit(0), "hipInit")

    module = ctypes.c_void_p()
    blob = (ctypes.c_char * len(hsaco)).from_buffer_copy(hsaco)
    hip.check(hip.lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(blob, ctypes.c_void_p)), "hipModuleLoadData")
    try:
        function = ctypes.c_void_p()
        hip.check(hip.lib.hipModuleGetFunction(ctypes.byref(function), module, KERNEL.encode()), "hipModuleGetFunction")

        sizes = [0, 1, WAVE_SIZE - 1, WAVE_SIZE, WAVE_SIZE + 7, 2 * WORKGROUP_SIZE + 3]
        elem_count = ((max(sizes) + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE) * WORKGROUP_SIZE
        rng = random.Random(37)
        x = [float(rng.randint(-17, 17)) for _ in range(elem_count)]
        y0 = [float(rng.randint(-23, 23)) for _ in range(elem_count)]
        byte_count = elem_count * ctypes.sizeof(ctypes.c_float)

        device_x = ctypes.c_void_p()
        device_y = ctypes.c_void_p()
        hip.check(hip.lib.hipMalloc(ctypes.byref(device_x), byte_count), "hipMalloc x")
        hip.check(hip.lib.hipMalloc(ctypes.byref(device_y), byte_count), "hipMalloc y")
        try:
            xb = (ctypes.c_float * elem_count)(*x)
            hip.check(
                hip.lib.hipMemcpy(device_x, ctypes.cast(xb, ctypes.c_void_p), byte_count, HIP_MEMCPY_HOST_TO_DEVICE),
                "memcpy x")
            grid_x = elem_count // WORKGROUP_SIZE
            for n in sizes:
                yb = (ctypes.c_float * elem_count)(*y0)
                hip.check(
                    hip.lib.hipMemcpy(device_y, ctypes.cast(yb, ctypes.c_void_p), byte_count,
                                      HIP_MEMCPY_HOST_TO_DEVICE), "memcpy y")
                x_arg = ctypes.c_void_p(device_x.value)
                y_arg = ctypes.c_void_p(device_y.value)
                a_arg = ctypes.c_float(ALPHA)
                n_arg = ctypes.c_uint32(n)
                params = (ctypes.c_void_p * 4)(_ptr_to(x_arg), _ptr_to(y_arg), _ptr_to(a_arg), _ptr_to(n_arg))
                hip.check(
                    hip.lib.hipModuleLaunchKernel(function, grid_x, 1, 1, WORKGROUP_SIZE, 1, 1, 0, None, params, None),
                    "hipModuleLaunchKernel")
                hip.check(hip.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
                got = (ctypes.c_float * elem_count)()
                hip.check(
                    hip.lib.hipMemcpy(ctypes.cast(got, ctypes.c_void_p), device_y, byte_count,
                                      HIP_MEMCPY_DEVICE_TO_HOST), "memcpy back")
                for i in range(elem_count):
                    expected = y0[i] + ALPHA * x[i] if i < n else y0[i]
                    if abs(got[i] - expected) > 1e-3:
                        raise AssertionError(f"n={n} i={i} expected={expected} got={got[i]}")
                print(f"  n={n} ok")
        finally:
            if device_x.value:
                hip.lib.hipFree(device_x)
            if device_y.value:
                hip.lib.hipFree(device_y)
    finally:
        hip.lib.hipModuleUnload(module)


def main(argv: list[str]) -> int:
    # Launch sub-mode: HIP-only, and must NOT import triton. Importing triton's
    # _C in-process breaks a separately-CDLL'd HIP runtime (-> hipErrorNoDevice),
    # so emit/assemble (triton) and the raw HIP launch never share a process.
    # (M3 launches via triton's own HIP driver instead of a raw CDLL.)
    if argv[:1] == ["_launch"]:
        launch_and_check(Path(argv[1]).read_bytes())
        return 0

    print(f"[M0] chip={chip()} wave-translate={wave_translate_bin()}")
    asm = emit_asm(SAXPY_WAVE_MLIR)
    print(f"[M0] wave-translate -> {len(asm.splitlines())} lines of AMDGPU asm")
    hsaco = assemble_link(asm)
    print(f"[M0] triton assemble+link -> {len(hsaco)} byte HSACO")
    print("[M0] launching on GPU (isolated subprocess):")
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco)
        hsaco_path = f.name
    try:
        lib_dir = str(Path(hip_lib()).parent)
        ld = os.pathsep.join(p for p in [lib_dir, os.environ.get("LD_LIBRARY_PATH", "")] if p)
        env = dict(os.environ, LD_LIBRARY_PATH=ld)
        subprocess.run([sys.executable, __file__, "_launch", hsaco_path], env=env, check=True)
    finally:
        os.unlink(hsaco_path)
    print("[M0] PASS: emit -> assemble -> link -> launch verified end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

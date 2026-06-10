#!/usr/bin/env python3
"""[M3] whole-pipeline end-to-end test for the wave_amd backend.

Compiles the 1-D masked-copy kernel through the REAL ``WaveAMDBackend`` -- the
full Wave path: finalized TTGIR -> Wave dialect (``make_wave``) -> AMDGPU asm
(``make_amdgcn`` / wave-translate) -> HSACO (inherited ``make_hsaco``) -- and
then LAUNCHES the resulting kernel on the gfx1100 GPU, checking numerics against
a host reference for a spread of sizes (boundary, sub-wave, multi-workgroup).

This is the capstone: it proves the converter is wired into the backend's stage
graph and that its output is a launchable, numerically-correct GPU kernel, with
both statically-linked LLVM runtimes in play (wave's via the wave-translate
subprocess inside ``make_amdgcn``; triton's via ``_C`` for assemble + link).

The compile (which imports triton's ``_C``) runs in the parent process; the
raw-HIP launch runs in a ``_launch`` SUBPROCESS that never imports triton --
importing triton's ``_C`` LLVM in-process breaks a separately-CDLL'd HIP runtime
(-> hipErrorNoDevice). This mirrors M0's isolation structure exactly.

Usage:
    python third_party/wave_amd/test/m3_end_to_end.py

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

KERNEL = "masked_copy"
BLOCK = 32
WAVE_SIZE = 32  # gfx11 wavefront
WORKGROUP_SIZE = 32  # one workgroup = one wave handles 32 consecutive elements

HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


def chip() -> str:
    return os.environ.get("WAVE_AMD_CHIP", "gfx1100")


def hip_lib() -> str:
    return os.environ.get("HIP_LIB", "/opt/rocm-6.3.3/lib/libamdhip64.so")


def compile_masked_copy() -> bytes:
    """Compile masked-copy through the real WaveAMDBackend; return the HSACO.

    Drives the same backend instance through the whole Wave path and the
    inherited assemble+link. Imports triton's ``_C`` (LLVM) -- so this must run
    ONLY in the parent process, never in the raw-HIP launch subprocess.
    """
    import triton
    import triton.language as tl
    from triton.backends.wave_amd.compiler import WaveAMDBackend
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir

    # WaveAMDBackend.supports_target gates on backend == "wave_amd" (the real
    # backend-discovery name), so the GPUTarget must carry that backend string;
    # the inherited HIP pipeline keys off target.arch, not the backend string.
    target = GPUTarget("wave_amd", chip(), WAVE_SIZE)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})

    @triton.jit
    def masked_copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, v, mask=mask)

    src = ASTSource(
        fn=masked_copy,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "n": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": BLOCK},
    )

    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()

    md = {}
    m = src.make_ir(target, options, codegen_fns, module_map, ctx)
    m = backend.make_ttir(m, md, options)
    m = backend.make_ttgir(m, md, options)

    # Wave path, driven on the SAME backend instance.
    wave_text = backend.make_wave(m, md, options)
    asm = backend.make_amdgcn(wave_text, md, options)

    from triton._C.libtriton import llvm
    llvm.init_targets()
    hsaco = backend.make_hsaco(asm, md, options)

    print(f"[M3] kernel name: {md.get('name')!r}")
    print(f"[M3] wave-translate -> {len(asm.splitlines())} lines of AMDGPU asm")
    print(f"[M3] assemble+link -> {len(hsaco)} byte HSACO")
    if md.get("name") != KERNEL:
        raise SystemExit(f"[M3] FAIL: expected kernel name {KERNEL!r}, got {md.get('name')!r}")
    if hsaco[:4] != b"\x7fELF":
        raise SystemExit("[M3] FAIL: produced HSACO is not an ELF object")
    return hsaco


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

        sizes = [0, 1, 31, 32, 39, 67, 2 * 32 + 3]
        elem_count = ((max(sizes) + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE) * WORKGROUP_SIZE
        # Disjoint deterministic fills: a copied lane (got == x) is unmistakable
        # from an untouched lane (got == y0) since the ranges never overlap.
        rng_x = random.Random(41)
        rng_y = random.Random(97)
        x = [float(rng_x.randint(-17, 17)) for _ in range(elem_count)]
        y0 = [float(rng_y.randint(1000, 2000)) for _ in range(elem_count)]
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
                # Kernarg ABI (24 bytes): x_ptr @0 (8), y_ptr @8 (8), n:i32 @16 (4).
                x_arg = ctypes.c_void_p(device_x.value)
                y_arg = ctypes.c_void_p(device_y.value)
                n_arg = ctypes.c_uint32(n)
                params = (ctypes.c_void_p * 3)(_ptr_to(x_arg), _ptr_to(y_arg), _ptr_to(n_arg))
                hip.check(
                    hip.lib.hipModuleLaunchKernel(function, grid_x, 1, 1, WORKGROUP_SIZE, 1, 1, 0, None, params, None),
                    "hipModuleLaunchKernel")
                hip.check(hip.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
                got = (ctypes.c_float * elem_count)()
                hip.check(
                    hip.lib.hipMemcpy(ctypes.cast(got, ctypes.c_void_p), device_y, byte_count,
                                      HIP_MEMCPY_DEVICE_TO_HOST), "memcpy back")
                for i in range(elem_count):
                    expected = x[i] if i < n else y0[i]
                    if got[i] != expected:
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
    # Launch sub-mode: raw HIP only, and must NOT import triton. Importing
    # triton's _C in-process breaks a separately-CDLL'd HIP runtime
    # (-> hipErrorNoDevice), so the parent (triton compile) and the raw-HIP
    # launch never share a process -- exactly as in M0.
    if argv[:1] == ["_launch"]:
        launch_and_check(Path(argv[1]).read_bytes())
        return 0

    print(f"[M3] chip={chip()} compiling masked-copy through WaveAMDBackend ...")
    hsaco = compile_masked_copy()
    print("[M3] launching on GPU (isolated subprocess):")
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco)
        hsaco_path = f.name
    try:
        lib_dir = str(Path(hip_lib()).parent)
        ld = os.pathsep.join(p for p in [lib_dir, os.environ.get("LD_LIBRARY_PATH", "")] if p)
        env = dict(os.environ, LD_LIBRARY_PATH=ld)
        sys.stdout.flush()  # keep parent banner ahead of the subprocess's per-n lines
        subprocess.run([sys.executable, __file__, "_launch", hsaco_path], env=env, check=True)
    finally:
        os.unlink(hsaco_path)
    print("[M3] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""[generalize stage 2] end-to-end strided 2-D copy on gfx1100.

Exercises the generalized converter's multi-register addressing -- a single op
class beyond the M2/M3 single-element-per-lane masked copy:

  * a 32x8 tile whose access layout spreads BOTH dims across register AND lane
    bits, bit-interleaved (lane=[[0,1],[0,2],[0,4],[1,0],[2,0]],
    reg=[[4,0],[8,0],[16,0]]) -- so per-lane offsets need shift/mask/mul/add,
    not just ``c * lane``;
  * a RUNTIME row stride (``stride_m`` kernarg), proving strides flow from the
    addptr DAG into the synthesized address, not just compile-time constants;
  * NO mask and NO convert_layout (those are the masked-copy path and stage 3).

Compiles the kernel through the REAL ``WaveAMDBackend`` (make_wave ->
wave_converter.convert) and LAUNCHES it, checking ``y == x`` on exactly the
strided footprint for several runtime stride values (contiguous and gapped).
Same process isolation as m3: compile in the parent, raw-HIP launch in a
``_launch`` subprocess that never imports triton.

Usage:
    PYTHONPATH=/home/vano/triton/python TRITON_BACKENDS_IN_TREE=1 \
        python third_party/wave_amd/test/strided_copy_e2e.py
Env: WAVE_REPO, WAVE_TRANSLATE, HIP_LIB, WAVE_AMD_CHIP (see m3_end_to_end.py).
"""

from __future__ import annotations

import ctypes
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = "strided_copy"
BM = 32  # rows  (spread across register bits + high lane bits)
BN = 8  # cols  (spread across low lane bits)
WAVE_SIZE = 32
STRIDES = [BN, 10, 13]  # contiguous, then gapped runtime strides

HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


def chip() -> str:
    return os.environ.get("WAVE_AMD_CHIP", "gfx1100")


def hip_lib() -> str:
    return os.environ.get("HIP_LIB", "/opt/rocm-6.3.3/lib/libamdhip64.so")


def compile_strided_copy() -> bytes:
    """Compile the strided copy through the real WaveAMDBackend; return HSACO."""
    import triton
    import triton.language as tl
    from triton.backends.wave_amd.compiler import WaveAMDBackend
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir, llvm

    target = GPUTarget("wave_amd", chip(), WAVE_SIZE)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": 1})

    @triton.jit
    def strided_copy(x_ptr, y_ptr, stride_m, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        offs = rm[:, None] * stride_m + rn[None, :]
        v = tl.load(x_ptr + offs)
        tl.store(y_ptr + offs, v)

    src = ASTSource(
        fn=strided_copy,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "stride_m": "i32", "BM": "constexpr", "BN": "constexpr"},
        constexprs={"BM": BM, "BN": BN},
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
    wave_text = backend.make_wave(m, md, options)
    asm = backend.make_amdgcn(wave_text, md, options)

    llvm.init_targets()
    hsaco = backend.make_hsaco(asm, md, options)
    print(f"[strided] kernel={md.get('name')!r}  asm={len(asm.splitlines())} lines  HSACO={len(hsaco)} bytes")
    if md.get("name") != KERNEL:
        raise SystemExit(f"[strided] FAIL: expected kernel {KERNEL!r}, got {md.get('name')!r}")
    if hsaco[:4] != b"\x7fELF":
        raise SystemExit("[strided] FAIL: produced HSACO is not an ELF object")
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

        max_stride = max(STRIDES)
        elem_count = BM * max_stride  # holds the largest strided footprint
        byte_count = elem_count * ctypes.sizeof(ctypes.c_float)
        rng_x = random.Random(7)
        rng_y = random.Random(23)
        # Disjoint ranges: a copied element (== x) is unmistakable from an
        # untouched one (== y0); the strided footprint leaves gaps untouched.
        x = [float(rng_x.randint(-500, 500)) for _ in range(elem_count)]
        y0 = [float(rng_y.randint(10_000, 20_000)) for _ in range(elem_count)]

        device_x = ctypes.c_void_p()
        device_y = ctypes.c_void_p()
        hip.check(hip.lib.hipMalloc(ctypes.byref(device_x), byte_count), "hipMalloc x")
        hip.check(hip.lib.hipMalloc(ctypes.byref(device_y), byte_count), "hipMalloc y")
        try:
            xb = (ctypes.c_float * elem_count)(*x)
            hip.check(
                hip.lib.hipMemcpy(device_x, ctypes.cast(xb, ctypes.c_void_p), byte_count, HIP_MEMCPY_HOST_TO_DEVICE),
                "memcpy x")
            for stride_m in STRIDES:
                yb = (ctypes.c_float * elem_count)(*y0)
                hip.check(
                    hip.lib.hipMemcpy(device_y, ctypes.cast(yb, ctypes.c_void_p), byte_count,
                                      HIP_MEMCPY_HOST_TO_DEVICE), "memcpy y")
                # Kernarg ABI: x_ptr @0 (8), y_ptr @8 (8), stride_m:i32 @16 (4).
                x_arg = ctypes.c_void_p(device_x.value)
                y_arg = ctypes.c_void_p(device_y.value)
                s_arg = ctypes.c_uint32(stride_m)
                params = (ctypes.c_void_p * 3)(_ptr_to(x_arg), _ptr_to(y_arg), _ptr_to(s_arg))
                # grid (1,1,1), block (WAVE_SIZE,1,1): one wave owns the 32x8 tile.
                hip.check(hip.lib.hipModuleLaunchKernel(function, 1, 1, 1, WAVE_SIZE, 1, 1, 0, None, params, None),
                          "hipModuleLaunchKernel")
                hip.check(hip.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
                got = (ctypes.c_float * elem_count)()
                hip.check(
                    hip.lib.hipMemcpy(ctypes.cast(got, ctypes.c_void_p), device_y, byte_count,
                                      HIP_MEMCPY_DEVICE_TO_HOST), "memcpy back")
                touched = {m * stride_m + n for m in range(BM) for n in range(BN)}
                for i in range(elem_count):
                    expected = x[i] if i in touched else y0[i]
                    if got[i] != expected:
                        m, n = divmod(i, stride_m)
                        raise AssertionError(
                            f"stride_m={stride_m} i={i} (m={m},n={n}) expected={expected} got={got[i]}")
                print(f"  stride_m={stride_m} ok ({len(touched)} elems copied)")
        finally:
            if device_x.value:
                hip.lib.hipFree(device_x)
            if device_y.value:
                hip.lib.hipFree(device_y)
    finally:
        hip.lib.hipModuleUnload(module)


def main(argv: list[str]) -> int:
    if argv[:1] == ["_launch"]:
        launch_and_check(Path(argv[1]).read_bytes())
        return 0

    print(f"[strided] chip={chip()} compiling {BM}x{BN} strided copy through WaveAMDBackend ...")
    hsaco = compile_strided_copy()
    print("[strided] launching on GPU (isolated subprocess):")
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco)
        hsaco_path = f.name
    try:
        lib_dir = str(Path(hip_lib()).parent)
        ld = os.pathsep.join(p for p in [lib_dir, os.environ.get("LD_LIBRARY_PATH", "")] if p)
        env = dict(os.environ, LD_LIBRARY_PATH=ld)
        sys.stdout.flush()
        subprocess.run([sys.executable, __file__, "_launch", hsaco_path], env=env, check=True)
    finally:
        os.unlink(hsaco_path)
    print("[strided] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

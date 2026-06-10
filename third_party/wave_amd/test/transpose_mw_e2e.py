#!/usr/bin/env python3
"""[generalize stage 4] multi-warp transpose through the converter, on gfx1100.

The single-warp transpose (transpose_e2e.py) keeps the whole convert_layout
inside one wave. This test runs the SAME kernel at ``num_warps=2``, where the
convert delta moves the WARP dimension (src warp -> dim1, dst warp -> dim0), so
the LDS roundtrip must shuffle data ACROSS warps and rely on the
workgroup-wide ``s_barrier``. It exercises:

  * warp bases in the access + convert layouts (``warp_id`` in synthesized
    offsets, from ``workitem_id >> log2(W)``);
  * a full-workgroup LDS arena (every warp's slots, row-major);
  * cross-warp correctness -- a missing/insufficient barrier would race and the
    distinct-value transpose check would catch it.

Two tile sizes (32x32 and 64x64) at num_warps=2. Block dim is
``num_warps * WAVE_SIZE``. Same process isolation as m3: compile in the parent,
raw-HIP launch in a ``_launch`` subprocess that never imports triton.

Usage:
    PYTHONPATH=/home/vano/triton/python TRITON_BACKENDS_IN_TREE=1 \
        python third_party/wave_amd/test/transpose_mw_e2e.py
Env: WAVE_REPO, WAVE_TRANSLATE, HIP_LIB, WAVE_AMD_CHIP (see m3_end_to_end.py).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = "transpose"
WAVE_SIZE = 32
NUM_WARPS = 2
CASES = [32, 64]  # square tile sizes, each at NUM_WARPS

HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


def chip() -> str:
    return os.environ.get("WAVE_AMD_CHIP", "gfx1100")


def hip_lib() -> str:
    return os.environ.get("HIP_LIB", "/opt/rocm-6.3.3/lib/libamdhip64.so")


def compile_transpose(n: int) -> bytes:
    """Compile an NxN transpose at num_warps=2 through the real backend (parent)."""
    import triton
    import triton.language as tl
    from triton.backends.wave_amd.compiler import WaveAMDBackend
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from triton._C.libtriton import ir, llvm

    target = GPUTarget("wave_amd", chip(), WAVE_SIZE)
    backend = WaveAMDBackend(target)
    options = backend.parse_options({"num_warps": NUM_WARPS})

    @triton.jit
    def transpose(x_ptr, y_ptr, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        x = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        xt = tl.trans(x)
        tl.store(y_ptr + rn[:, None] * BM + rm[None, :], xt)

    src = ASTSource(
        fn=transpose,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "BM": "constexpr", "BN": "constexpr"},
        constexprs={"BM": n, "BN": n},
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
    print(f"[transpose-mw {n}x{n} w{NUM_WARPS}] barriers={wave_text.count('wave.barrier')} "
          f"-> s_barrier={asm.count('s_barrier')} ds_store={asm.count('ds_store')} ds_load={asm.count('ds_load')}")

    llvm.init_targets()
    hsaco = backend.make_hsaco(asm, md, options)
    if md.get("name") != KERNEL:
        raise SystemExit(f"[transpose-mw] FAIL: expected kernel {KERNEL!r}, got {md.get('name')!r}")
    if hsaco[:4] != b"\x7fELF":
        raise SystemExit("[transpose-mw] FAIL: produced HSACO is not an ELF object")
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


def launch_and_check(hip: "Hip", function, n: int) -> None:
    count = n * n
    x = [float(i * 1000 + j) for i in range(n) for j in range(n)]  # distinct
    expected = [x[j * n + i] for i in range(n) for j in range(n)]  # y = x^T
    byte_count = count * ctypes.sizeof(ctypes.c_float)

    device_x = ctypes.c_void_p()
    device_y = ctypes.c_void_p()
    hip.check(hip.lib.hipMalloc(ctypes.byref(device_x), byte_count), "hipMalloc x")
    hip.check(hip.lib.hipMalloc(ctypes.byref(device_y), byte_count), "hipMalloc y")
    try:
        xb = (ctypes.c_float * count)(*x)
        hip.check(hip.lib.hipMemcpy(device_x, ctypes.cast(xb, ctypes.c_void_p), byte_count, HIP_MEMCPY_HOST_TO_DEVICE),
                  "memcpy x")
        yb = (ctypes.c_float * count)(*([-1.0] * count))
        hip.check(hip.lib.hipMemcpy(device_y, ctypes.cast(yb, ctypes.c_void_p), byte_count, HIP_MEMCPY_HOST_TO_DEVICE),
                  "memcpy y init")
        x_arg = ctypes.c_void_p(device_x.value)
        y_arg = ctypes.c_void_p(device_y.value)
        params = (ctypes.c_void_p * 2)(_ptr_to(x_arg), _ptr_to(y_arg))
        block = NUM_WARPS * WAVE_SIZE
        hip.check(hip.lib.hipModuleLaunchKernel(function, 1, 1, 1, block, 1, 1, 0, None, params, None),
                  "hipModuleLaunchKernel")
        hip.check(hip.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
        got = (ctypes.c_float * count)()
        hip.check(hip.lib.hipMemcpy(ctypes.cast(got, ctypes.c_void_p), device_y, byte_count, HIP_MEMCPY_DEVICE_TO_HOST),
                  "memcpy back")
        for idx in range(count):
            if got[idx] != expected[idx]:
                i, j = divmod(idx, n)
                raise AssertionError(f"{n}x{n}: mismatch at y[{i},{j}] expected {expected[idx]} got {got[idx]}")
        print(f"  transpose {n}x{n} w{NUM_WARPS} (block={NUM_WARPS * WAVE_SIZE}): all {count} elements match x^T")
    finally:
        if device_x.value:
            hip.lib.hipFree(device_x)
        if device_y.value:
            hip.lib.hipFree(device_y)


def launch_case(hsaco: bytes, n: int) -> None:
    hip = Hip(hip_lib())
    hip.check(hip.lib.hipInit(0), "hipInit")
    module = ctypes.c_void_p()
    blob = (ctypes.c_char * len(hsaco)).from_buffer_copy(hsaco)
    hip.check(hip.lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(blob, ctypes.c_void_p)), "hipModuleLoadData")
    try:
        function = ctypes.c_void_p()
        hip.check(hip.lib.hipModuleGetFunction(ctypes.byref(function), module, KERNEL.encode()), "hipModuleGetFunction")
        launch_and_check(hip, function, n)
    finally:
        hip.lib.hipModuleUnload(module)


def main(argv: list[str]) -> int:
    if argv[:1] == ["_launch"]:
        launch_case(Path(argv[1]).read_bytes(), int(argv[2]))
        return 0

    print(f"[transpose-mw] chip={chip()} compiling transpose at num_warps={NUM_WARPS} ...")
    for n in CASES:
        hsaco = compile_transpose(n)
        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
            f.write(hsaco)
            hsaco_path = f.name
        try:
            lib_dir = str(Path(hip_lib()).parent)
            ld = os.pathsep.join(p for p in [lib_dir, os.environ.get("LD_LIBRARY_PATH", "")] if p)
            env = dict(os.environ, LD_LIBRARY_PATH=ld)
            sys.stdout.flush()
            subprocess.run([sys.executable, __file__, "_launch", hsaco_path, str(n)], env=env, check=True)
        finally:
            os.unlink(hsaco_path)
    print("[transpose-mw] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""[M2b] emit AMDGPU asm from the M2 converter output via wave-translate.

Chains the front of the wave_amd pipeline that now exists:

    masked-copy kernel
      -> finalized TTGIR            (triton's AMD make_ttir/make_ttgir, as in m1b)
      -> Wave dialect               (M2 structural converter)
      -> AMDGPU asm                 (wave-translate --wave-to-amdgpu-asm, as in M0)

This is the in-repo bridge between M2 (structural conversion) and the back half
proven in M0 (asm -> assemble -> link -> launch). It confirms the converter's
Wave IR is not just verifiable but *lowerable*: wave-translate must accept every
op/type/predicate the converter emits and produce a kernel with the right
symbol, EXEC-mask predication (from wave.where), the masked global load/store,
and a kernarg ABI matching the kernel signature. No GPU, no launch here -- that
is M3.

Usage: python third_party/wave_amd/test/m2b_emit.py
Env:   WAVE_REPO       wave-mlir repo root    (default: /home/vano/7/7)
       WAVE_TRANSLATE  wave-translate path    (default: $WAVE_REPO/build/bin/wave-translate)
"""

from __future__ import annotations

import sys
from pathlib import Path

ARCH = "gfx1100"

_TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TEST_DIR))
sys.path.insert(0, str(_TEST_DIR.parent / "backend"))


def main() -> int:
    # m2_converter compiles the masked-copy kernel to finalized TTGIR; m0 owns
    # the wave-translate invocation; wave_converter is the M2 bridge between them.
    from m2_converter import build_finalized_ttgir
    from wave_converter import convert, wave_ir_text
    from m0_back_half_smoke import emit_asm, wave_translate_bin

    print(f"[M2b] masked-copy -> finalized TTGIR -> Wave dialect ({ARCH}) ...")
    ttgir_module, _context = build_finalized_ttgir()
    bld = convert(ttgir_module, ARCH)
    wave_text = wave_ir_text(bld)
    if not bld.module.operation.verify():
        sys.stderr.write(wave_text + "\n")
        raise SystemExit("[M2b] FAIL: converted Wave module did not verify")

    print(f"[M2b] wave-translate {wave_translate_bin()} --wave-to-amdgpu-asm ...")
    asm = emit_asm(wave_text)
    print(f"[M2b]   -> {len(asm.splitlines())} lines of AMDGPU asm")

    # The asm must be a real gfx1100 kernel object for `masked_copy`, with the
    # mask lowered to EXEC predication, the masked memory ops present, and a
    # kernarg block describing (x_ptr, y_ptr, n).
    required = [
        f'.amdgcn_target "amdgcn-amd-amdhsa--{ARCH}"', "masked_copy:", ".amdhsa_kernel masked_copy",
        "s_and_saveexec",  # wave.where -> EXEC mask
        "s_cbranch_execz",  # skip the masked body when no lane is active
        "global_load",  # masked load
        "global_store",  # masked store
        "s_endpgm", ".amdgpu_metadata", "global_buffer",  # x_ptr / y_ptr kernargs
    ]
    missing = [tok for tok in required if tok not in asm]
    if missing:
        sys.stderr.write(asm + "\n")
        raise SystemExit(f"[M2b] FAIL: AMDGPU asm missing expected markers: {missing}")

    # Kernarg size must match the signature: 2 pointers (8B each) + i32 (4B) = 24.
    if ".amdhsa_kernarg_size 24" not in asm:
        sys.stderr.write(asm + "\n")
        raise SystemExit("[M2b] FAIL: expected 24-byte kernarg block (x_ptr, y_ptr, n)")

    print("[M2b] PASS: converter Wave IR lowers to a complete gfx1100 masked_copy kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

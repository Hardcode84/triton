import functools

from triton.backends.compiler import GPUTarget, Language
from triton.backends.amd import compiler as amd_compiler
from triton import knobs


class WaveAMDBackend(amd_compiler.HIPBackend):
    """Triton backend for AMD GPUs that lowers through the Wave dialect.

    This backend deliberately reuses almost all of Triton's upstream AMD
    (HIP) backend. By subclassing ``HIPBackend`` we inherit the real
    ``make_ttir`` / ``make_ttgir`` pipeline -- including the full AMD layout
    pipeline (convert-to-ttgpuir, coalesce, remove-layout-conversions,
    accelerate-matmul, etc.) -- and the ``make_hsaco`` step (which runs
    ``amd.assemble_amdgcn`` + ``amd.link_hsaco``) verbatim. The only
    backend-specific behavior is a Wave-dialect path that replaces the HIP
    backend's ``llir`` -> ``amdgcn`` stages with new ``wave`` and ``amdgcn``
    stages:

      * ``make_wave``: finalized TTGIR -> Wave dialect (lands in M2).
      * ``make_amdgcn``: Wave dialect -> AMDGPU asm via wave-translate (M2b).

    Both Wave-path stages are stubbed (``NotImplementedError``) in M1; the
    skeleton exists so backend discovery, gating, and the stage graph can be
    validated before the Wave lowering itself is implemented.
    """

    @staticmethod
    def supports_target(target):
        return target.backend == "wave_amd"

    def __init__(self, target):
        super().__init__(target)
        if not str(target.arch).startswith("gfx"):
            raise ValueError(f"wave_amd backend requires a gfx arch, got {target.arch!r}")
        self.binary_ext = "hsaco"

    def get_target_name(self, options) -> str:
        return f"wave_amd:{options.arch}"

    @staticmethod
    def make_wave(src, metadata, options):
        # src: finalized TTGIR ModuleOp. Convert structurally to Wave dialect IR
        # text (the converter walks TTGIR via triton's _C bindings and emits Wave
        # IR via wave-mlir's bindings in-process). Returns Wave IR text because
        # the next stage runs in wave-mlir's separate LLVM runtime.
        from triton.backends.wave_amd.wave_converter import convert, wave_ir_text
        metadata["name"] = src.get_entry_func_name()
        bld = convert(src, options.arch)
        if not bld.module.operation.verify():
            raise RuntimeError("wave_amd: converted Wave module failed to verify")
        return wave_ir_text(bld)

    @staticmethod
    def make_amdgcn(src, metadata, options):
        # src: high-level Wave IR text. Lower to AMDGPU asm via wave-translate
        # (wave-mlir's LLVM). The inherited make_hsaco then assembles + links it
        # with triton's LLVM, exactly as the HIP backend does for its own asm.
        from triton.backends.wave_amd.wave_converter import wave_to_amdgpu_asm
        return wave_to_amdgpu_asm(src)

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("wave_amd only supports Triton (TTIR) input")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["wave"] = lambda src, metadata: self.make_wave(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        return f"{self.target}-wave_amd"

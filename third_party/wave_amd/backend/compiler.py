import functools
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

from triton import knobs
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.backends.wave_amd.emission import emit_amdgcn_from_wave_mlir, emit_hsaco_from_amdgcn, _packaged_wave_translate
from triton.backends.wave_amd.lowering import lower_ttir_to_wave_mlir
from triton.backends.wave_amd.pipeline import clone_module_for_preview, run_wave_ttgir_preview_pipeline, \
    run_wave_ttir_pipeline


def _warp_size_for_arch(arch: str) -> int:
    match = re.match(r"^gfx(\d+)", arch)
    if match is None:
        raise ValueError(f"wave_amd expects an AMDGPU gfx architecture, got {arch!r}")

    digits = match.group(1)
    gfx_major = int(digits[:-2]) if len(digits) >= 4 else int(digits[0])
    return 32 if gfx_major >= 10 else 64


def _min_dot_size(target: GPUTarget):
    return lambda lhs_type, rhs_type: (1, 1, 1)


def _wave_backend_artifact_paths() -> Tuple[Path, ...]:
    backend_dir = Path(__file__).resolve().parent
    return (
        _packaged_wave_translate(),
        backend_dir / "share" / "wave-mlir" / "pipelines" / "pipelines.mlir",
    )


def _wave_backend_artifact_hash() -> str:
    digest = hashlib.sha256()
    for path in _wave_backend_artifact_paths():
        digest.update(str(path.name).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(f"missing:{path}".encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class WaveAMDOptions:
    num_warps: int = 4
    waves_per_eu: int = 0
    num_stages: int = 2
    num_ctas: int = 1
    extern_libs: dict = None
    debug: bool = False
    instrumentation_mode: str = ""
    sanitize_overflow: bool = True
    arch: str = None
    supported_fp8_dtypes: Tuple[str] = ("fp8e4nv", "fp8e5", "fp8e5b16", "fp8e4b8")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "bf16x3", "bf16x6")
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    enable_ttgir_preview: bool = False
    backend_name: str = "wave_amd"

    def __post_init__(self):
        assert self.arch is not None
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, "extern_libs", tuple(sorted(extern_libs.items())))
        object.__setattr__(self, "warp_size", _warp_size_for_arch(self.arch))

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in sorted(self.__dict__.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class WaveAMDBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "wave_amd"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        assert isinstance(target.arch, str)
        self.binary_ext = "hsaco"

    def get_target_name(self, options) -> str:
        return f"wave_amd:{options.arch}"

    def parse_options(self, opts) -> Any:
        args = {"arch": knobs.runtime.override_arch or self.target.arch}
        args.update(
            {k: opts[k]
             for k in WaveAMDOptions.__dataclass_fields__.keys()
             if k in opts and opts[k] is not None})
        return WaveAMDOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            getattr(metadata, "shared", 0),
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": _min_dot_size(self.target)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        try:
            from triton.language.extra.hip import libdevice
        except ModuleNotFoundError:
            return {}
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        # Wave MLIR is built through the Wave Python bindings.
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        return run_wave_ttir_pipeline(mod)

    @staticmethod
    def make_wave(src, metadata, options):
        wave_mlir, name = lower_ttir_to_wave_mlir(src, options)
        metadata["name"] = name
        metadata["shared"] = 0
        metadata["global_scratch_size"] = 0
        metadata["global_scratch_align"] = 1
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1
        metadata["tensordesc_meta"] = {}
        return wave_mlir

    @staticmethod
    def make_ttgir_preview(src, metadata, options):
        preview = clone_module_for_preview(src)
        run_wave_ttgir_preview_pipeline(preview, options)
        metadata["wave_ttgir_preview"] = str(preview)
        return src

    @staticmethod
    def make_amdgcn(src, metadata, options):
        return emit_amdgcn_from_wave_mlir(src, options)

    @staticmethod
    def make_hsaco(src, metadata, options):
        return emit_hsaco_from_amdgcn(src, options)

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("wave_amd only supports Triton TTIR input")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        if options.enable_ttgir_preview:
            stages["ttgir_preview"] = lambda src, metadata: self.make_ttgir_preview(src, metadata, options)
        stages["wave"] = lambda src, metadata: self.make_wave(src, metadata, options)
        stages["amdgcn"] = lambda src, metadata: self.make_amdgcn(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        return f"{self.target}-wave_amd-{_wave_backend_artifact_hash()}"

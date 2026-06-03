import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from triton import knobs

WAVE_TRANSLATE_RELATIVE_PATH = Path("bin") / "wave-translate"


def emit_amdgcn_from_wave_mlir(wave_mlir: str, options, wave_translate: Optional[Path] = None) -> str:
    """Emit AMDGCN assembly from Wave MLIR through the Wave translation tool."""

    if not isinstance(wave_mlir, str):
        raise TypeError("wave_amd AMDGCN emission expects Wave MLIR text")

    tool = _require_executable(Path(wave_translate) if wave_translate is not None else _packaged_wave_translate())
    proc = subprocess.run(
        [str(tool), "--wave-to-amdgpu-asm", "-"],
        input=wave_mlir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"wave_amd AMDGCN emission failed for {options.arch} using {tool}{detail}")
    return proc.stdout


def emit_hsaco_from_amdgcn(amdgcn: str, options, amd_codegen=None) -> bytes:
    """Assemble AMDGCN assembly and link it into HSACO bytes."""

    if not isinstance(amdgcn, str):
        raise TypeError("wave_amd HSACO emission expects AMDGCN assembly text")

    amd = amd_codegen if amd_codegen is not None else _triton_amd_codegen()
    target_features = "+xnack" if knobs.compilation.enable_asan else ""
    try:
        obj = amd.assemble_amdgcn(amdgcn, options.arch, target_features)
    except Exception as exc:
        raise RuntimeError(f"wave_amd HSACO emission failed while assembling AMDGCN for {options.arch} "
                           f"using Triton AMD codegen helpers: {exc}") from exc

    if not isinstance(obj, bytes):
        raise TypeError("wave_amd HSACO assembly helper must return bytes")

    try:
        with tempfile.NamedTemporaryFile() as tmp_out:
            with tempfile.NamedTemporaryFile() as tmp_in:
                with open(tmp_in.name, "wb") as fd_in:
                    fd_in.write(obj)
                amd.link_hsaco(tmp_in.name, tmp_out.name)
            with open(tmp_out.name, "rb") as fd_out:
                return fd_out.read()
    except Exception as exc:
        raise RuntimeError(f"wave_amd HSACO emission failed while linking AMDGCN object for {options.arch} "
                           f"using Triton AMD codegen helpers: {exc}") from exc


def _packaged_wave_translate() -> Path:
    return Path(__file__).resolve().parent / WAVE_TRANSLATE_RELATIVE_PATH


def _triton_amd_codegen():
    try:
        from triton._C.libtriton import amd, llvm
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("wave_amd HSACO emission requires Triton AMD codegen helpers "
                           "`assemble_amdgcn` and `link_hsaco`. Rebuild Triton with AMD backend support.") from exc

    llvm.init_targets()
    missing = [name for name in ("assemble_amdgcn", "link_hsaco") if not hasattr(amd, name)]
    if missing:
        raise RuntimeError("wave_amd HSACO emission requires Triton AMD codegen helpers "
                           f"{', '.join(missing)}. Rebuild Triton with AMD backend support.")
    return amd


def _require_executable(path: Path) -> Path:
    if _is_executable(path):
        return path
    raise RuntimeError("wave_amd AMDGCN emission requires the packaged Wave `wave-translate` tool. "
                       f"Expected executable at {path}. Rebuild Triton so the wave_amd backend CMake step builds and "
                       "packages wave-translate.")


def _is_executable(path: Path) -> bool:
    return path.is_file() and path.stat().st_mode & 0o111 != 0

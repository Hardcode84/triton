import subprocess
from pathlib import Path
from typing import Optional

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


def _packaged_wave_translate() -> Path:
    return Path(__file__).resolve().parent / WAVE_TRANSLATE_RELATIVE_PATH


def _require_executable(path: Path) -> Path:
    if _is_executable(path):
        return path
    raise RuntimeError("wave_amd AMDGCN emission requires the packaged Wave `wave-translate` tool. "
                       f"Expected executable at {path}. Rebuild Triton so the wave_amd backend CMake step builds and "
                       "packages wave-translate.")


def _is_executable(path: Path) -> bool:
    return path.is_file() and path.stat().st_mode & 0o111 != 0

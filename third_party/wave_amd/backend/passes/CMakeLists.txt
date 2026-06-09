# Wave AMD backend C++ MLIR preparation passes.
#
# These passes transform Triton TTIR/TTGIR dialect IR and therefore live in the
# Triton tree (built against Triton's LLVM), not in the Wave submodule. They are
# registered into triton-opt and exposed to the Python pipeline through the
# TritonWaveAMD plugin.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonWaveAMD)
add_public_tablegen_target(TritonWaveAMDBackendPassesIncGen)

add_triton_library(TritonWaveAMDBackendPasses
  ConvertToTTGPUIR.cpp
  LegalizeDots.cpp
  PlanBufferDescriptors.cpp
  PlanGemmSchedule.cpp

  DEPENDS
  TritonWaveAMDBackendPassesIncGen
  TritonIR
  TritonGPUIR

  LINK_LIBS PUBLIC
  TritonToTritonGPU
)

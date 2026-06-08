#ifndef WAVE_AMD_BACKEND_PASSES_PASSES_H
#define WAVE_AMD_BACKEND_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "wave_amd/backend/passes/Passes.h.inc"

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "wave_amd/backend/passes/Passes.h.inc"

} // namespace mlir

#endif // WAVE_AMD_BACKEND_PASSES_PASSES_H

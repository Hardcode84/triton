#ifndef TRITON_CONVERSION_PASSES_H
#define TRITON_CONVERSION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"

// Convert a Triton module to TritonGPU in place, assigning the default blocked
// layouts parameterized by the warp configuration. This is the body of the
// `convert-triton-to-tritongpu` pass exposed as a free function so a backend
// can reuse the exact conversion from its own pass instead of re-deriving it.
LogicalResult convertToTritonGPU(ModuleOp mod, StringRef target, int numWarps,
                                 int threadsPerWarp, int numCTAs,
                                 bool enableSourceRemat);

} // namespace mlir::triton

#endif

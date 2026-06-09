#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "TritonAMDGPUTransforms/Passes.h.inc"

void registerConSanAMDHooks();

} // namespace mlir

namespace mlir::triton::amdgpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL_TRITONAMDGPUOPTIMIZEDOTOPERANDS
#include "TritonAMDGPUTransforms/Passes.h.inc"

void registerTritonAMDGPUOptimizeDotOperands();

// Run the tritonamdgpu-accelerate-matmul transformation on a module in place.
// This is the body of the accelerate-matmul pass exposed as a free function so
// another backend (Wave AMD) can reuse the exact dot legalization, intrinsic
// legality, and architecture policy instead of re-deriving them.
LogicalResult accelerateMatmul(ModuleOp mod, StringRef gfxArch,
                               int matrixInstructionSize, int kPack);
} // namespace mlir::triton::amdgpu

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUTransforms/Passes.h.inc"
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_

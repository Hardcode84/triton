//===- AccelerateMatmul.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wave-owned AMD matrix-core dot legalization. It currently delegates to the
// base tritonamdgpu-accelerate-matmul transformation so behavior is identical,
// but it gives the Wave pipeline its own legalization seam to diverge toward
// Wave-native fragment layouts later, instead of calling the AMD pass directly.
//
//===----------------------------------------------------------------------===//

#include "wave_amd/backend/passes/Passes.h"

#include "TritonAMDGPUTransforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONWAVEAMDACCELERATEMATMUL
#include "wave_amd/backend/passes/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct TritonWaveAMDAccelerateMatmulPass
    : public impl::TritonWaveAMDAccelerateMatmulBase<
          TritonWaveAMDAccelerateMatmulPass> {
  using impl::TritonWaveAMDAccelerateMatmulBase<
      TritonWaveAMDAccelerateMatmulPass>::TritonWaveAMDAccelerateMatmulBase;

  void runOnOperation() override {
    if (failed(triton::amdgpu::accelerateMatmul(getOperation(), gfxArch,
                                                matrixInstructionSize, kPack)))
      return signalPassFailure();
  }
};

} // namespace

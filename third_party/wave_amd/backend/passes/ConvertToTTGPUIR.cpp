//===- ConvertToTTGPUIR.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wave-owned Triton-to-TritonGPU conversion. It currently delegates to the base
// conversion so behavior is byte-identical, but it gives the Wave pipeline its
// own conversion seam to diverge toward Wave-native layout contracts later,
// instead of calling `convert-triton-to-tritongpu` directly.
//
//===----------------------------------------------------------------------===//

#include "wave_amd/backend/passes/Passes.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONWAVEAMDCONVERTTOTTGPUIR
#include "wave_amd/backend/passes/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct TritonWaveAMDConvertToTTGPUIRPass
    : public impl::TritonWaveAMDConvertToTTGPUIRBase<
          TritonWaveAMDConvertToTTGPUIRPass> {
  using impl::TritonWaveAMDConvertToTTGPUIRBase<
      TritonWaveAMDConvertToTTGPUIRPass>::TritonWaveAMDConvertToTTGPUIRBase;

  void runOnOperation() override {
    if (target.empty()) {
      emitError(getOperation().getLoc(),
                "'tritonwaveamd-convert-to-ttgpuir' requires 'target' option "
                "to be set");
      return signalPassFailure();
    }

    if (failed(triton::convertToTritonGPU(getOperation(), target, numWarps,
                                          threadsPerWarp, numCTAs,
                                          enableSourceRemat)))
      return signalPassFailure();
  }
};

} // namespace

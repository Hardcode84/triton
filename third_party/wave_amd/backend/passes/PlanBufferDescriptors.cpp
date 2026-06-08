//===- PlanBufferDescriptors.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wave TTGIR GEMM preparation: decide buffer descriptor ranges for static
// footprints in TTGIR, replacing the post-bridge Wave-dialect materialization
// pass. The range is attached as a typed attribute that the bridge consumes
// mechanically when it emits Wave buffer descriptors.
//
//===----------------------------------------------------------------------===//

#include "wave_amd/backend/passes/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONWAVEAMDPLANBUFFERDESCRIPTORS
#include "wave_amd/backend/passes/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace tt = mlir::triton;

namespace {

constexpr StringLiteral kRangeBytesAttr = "waveamd.buffer.range_bytes";

// Byte width of the element types supported for static buffer planning. Mirrors
// gemm_pipeline.buffer_plan_for_static_footprint, which only plans for these.
static std::optional<int64_t> supportedByteWidth(Type type) {
  if (type.isF16() || type.isBF16())
    return 2;
  if (type.isF32())
    return 4;
  return std::nullopt;
}

// Range in bytes of a statically-shaped tensor footprint, or nullopt if the
// footprint is not static or the element type is unsupported.
static std::optional<int64_t> staticRangeBytes(Type valueType) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueType);
  if (!tensorTy || !tensorTy.hasStaticShape())
    return std::nullopt;
  std::optional<int64_t> byteWidth =
      supportedByteWidth(tensorTy.getElementType());
  if (!byteWidth)
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t dim : tensorTy.getShape()) {
    if (dim <= 0)
      return std::nullopt;
    elements *= dim;
  }
  return elements * *byteWidth;
}

struct TritonWaveAMDPlanBufferDescriptorsPass
    : public impl::TritonWaveAMDPlanBufferDescriptorsBase<
          TritonWaveAMDPlanBufferDescriptorsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Builder b(&getContext());

    auto annotate = [&](Operation *op, Type valueType) {
      if (op->hasAttr(kRangeBytesAttr))
        return;
      std::optional<int64_t> range = staticRangeBytes(valueType);
      if (range)
        op->setAttr(kRangeBytesAttr, b.getI32IntegerAttr(*range));
    };

    mod.walk([&](Operation *op) {
      if (auto load = dyn_cast<tt::LoadOp>(op))
        annotate(load, load.getResult().getType());
      else if (auto store = dyn_cast<tt::StoreOp>(op))
        annotate(store, store.getValue().getType());
    });
  }
};

} // namespace

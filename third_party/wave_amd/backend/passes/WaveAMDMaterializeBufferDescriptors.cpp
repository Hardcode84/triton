//===- WaveAMDMaterializeBufferDescriptors.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Staged outside the Wave submodule until this pass is ready to be upstreamed
// into third_party/wave_amd/wave/lib/Dialect/Wave/Transforms.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Wave/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Wave/IR/Wave.h"
#include "mlir/Dialect/Wave/IR/WaveAMD.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"

#include <limits>
#include <utility>

namespace mlir::wave {
#define GEN_PASS_DEF_WAVEAMDMATERIALIZEBUFFERDESCRIPTORS
#include "mlir/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace mlir::wave

using namespace mlir;
using namespace mlir::wave;

namespace {

constexpr StringLiteral kRangeBytesAttr = "waveamd.buffer.range_bytes";

static Type convertPointerTypeToBuffer(Type type) {
  MLIRContext *ctx = type.getContext();
  auto bufferAddressSpace = waveamd::BufferAddressSpaceAttr::get(ctx);
  if (auto ptrType = dyn_cast<PtrType>(type))
    return PtrType::get(ctx, ptrType.getElementType(), bufferAddressSpace);
  if (auto simdType = dyn_cast<SimdType>(type)) {
    Type elementType = convertPointerTypeToBuffer(simdType.getElementType());
    if (!elementType)
      return Type();
    return SimdType::get(ctx, elementType, simdType.getWidth());
  }
  return Type();
}

static FailureOr<PtrType> getGlobalBasePointerType(PtrAddOp op) {
  auto baseType = dyn_cast<PtrType>(op.getBase().getType());
  if (!baseType) {
    op.emitOpError(kRangeBytesAttr) << " requires a uniform wave pointer base";
    return failure();
  }
  if (!isa<GlobalAddressSpaceAttr>(baseType.getAddressSpace())) {
    op.emitOpError(kRangeBytesAttr) << " requires a global wave pointer base";
    return failure();
  }
  if (!baseType.getElementType()) {
    op.emitOpError(kRangeBytesAttr) << " requires a typed wave pointer base";
    return failure();
  }
  return baseType;
}

struct WaveAMDMaterializeBufferDescriptorsPass
    : public wave::impl::WaveAMDMaterializeBufferDescriptorsBase<
          WaveAMDMaterializeBufferDescriptorsPass> {
  void runOnOperation() override {
    SmallVector<PtrAddOp> annotatedOps;
    getOperation()->walk([&](PtrAddOp op) {
      if (op->hasAttr(kRangeBytesAttr))
        annotatedOps.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    llvm::DenseMap<Block *, llvm::DenseMap<std::pair<Value, int64_t>, Value>>
        buffersByBlock;

    for (PtrAddOp op : annotatedOps) {
      auto rangeAttr = op->getAttrOfType<IntegerAttr>(kRangeBytesAttr);
      if (!rangeAttr) {
        op.emitOpError(kRangeBytesAttr) << " must be an integer attribute";
        return signalPassFailure();
      }
      int64_t rangeBytes = rangeAttr.getInt();
      if (rangeBytes <= 0 || rangeBytes > std::numeric_limits<int32_t>::max()) {
        op.emitOpError(kRangeBytesAttr)
            << " must fit in a positive i32 byte range";
        return signalPassFailure();
      }

      FailureOr<PtrType> baseType = getGlobalBasePointerType(op);
      if (failed(baseType))
        return signalPassFailure();

      Type bufferPtrAddType = convertPointerTypeToBuffer(op.getType());
      if (!bufferPtrAddType) {
        op.emitOpError(kRangeBytesAttr)
            << " requires a pointer or SIMD-of-pointer result";
        return signalPassFailure();
      }

      Block *block = op->getBlock();
      auto key = std::make_pair(op.getBase(), rangeBytes);
      auto &blockBuffers = buffersByBlock[block];
      Value buffer = blockBuffers.lookup(key);
      if (!buffer) {
        Type bufferType =
            PtrType::get(&getContext(), baseType->getElementType(),
                         waveamd::BufferAddressSpaceAttr::get(&getContext()));
        rewriter.setInsertionPoint(op);
        Value range =
            rewriter.create<arith::ConstantIntOp>(op.getLoc(), rangeBytes, 32);
        buffer = rewriter.create<waveamd::MakeBufferOp>(op.getLoc(), bufferType,
                                                        op.getBase(), range);
        blockBuffers[key] = buffer;
      }

      rewriter.setInsertionPoint(op);
      auto replacement = rewriter.create<PtrAddOp>(
          op.getLoc(), bufferPtrAddType, buffer, op.getOffset());
      rewriter.replaceOp(op, replacement.getResult());
    }
  }
};

} // namespace

//===- LegalizeDots.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wave TTGIR GEMM preparation: legalize dots against Wave AMD architecture
// policy and emit typed preparation attributes that the Python bridge consumes
// mechanically instead of parsing TTGIR type strings.
//
//===----------------------------------------------------------------------===//

#include "wave_amd/backend/passes/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
#define GEN_PASS_DEF_TRITONWAVEAMDLEGALIZEDOTS
#include "wave_amd/backend/passes/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

constexpr StringLiteral kInstrKindAttr = "waveamd.dot.instr_kind";
constexpr StringLiteral kAOpIdxAttr = "waveamd.dot.a_op_idx";
constexpr StringLiteral kBOpIdxAttr = "waveamd.dot.b_op_idx";
constexpr StringLiteral kRoleAttr = "waveamd.dot.role";

// Short element-type name used in the instruction kind string.
static StringRef elementName(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (type.isInteger(8))
    return "i8";
  if (type.isInteger(32))
    return "i32";
  return "unk";
}

// An MFMA encoding either directly or as the parent of a dot-operand encoding.
static bool isMfmaEncoding(Attribute encoding) {
  if (!encoding)
    return false;
  if (isa<ttg::AMDMfmaEncodingAttr>(encoding))
    return true;
  if (auto dotOp = dyn_cast<ttg::DotOperandEncodingAttr>(encoding))
    return isa<ttg::AMDMfmaEncodingAttr>(dotOp.getParent());
  return false;
}

// Attach the typed dot plan attributes for a WMMA matrix-core dot.
static void annotateWmmaDot(tt::DotOp dot, ttg::AMDWmmaEncodingAttr wmma) {
  auto resTy = cast<RankedTensorType>(dot.getType());
  auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
  if (!aTy || !bTy)
    return;

  ArrayRef<unsigned> mnk = wmma.getInstrShape();
  if (mnk.size() != 3)
    return;

  std::string kind;
  llvm::raw_string_ostream os(kind);
  os << "wmma." << elementName(resTy.getElementType()) << "." << mnk[0] << "x"
     << mnk[1] << "x" << mnk[2] << "." << elementName(aTy.getElementType());

  Builder b(dot.getContext());
  dot->setAttr(kInstrKindAttr, b.getStringAttr(os.str()));

  // Tag the operand and its source convert_layout with the role, so the bridge
  // reads the role from a prepared attr instead of parsing the dot_op encoding.
  auto tagRole = [&](Value operand, ttg::DotOperandEncodingAttr enc) {
    if (auto cvt = operand.getDefiningOp<ttg::ConvertLayoutOp>())
      cvt->setAttr(kRoleAttr, b.getI32IntegerAttr(enc.getOpIdx()));
  };
  auto aDot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  auto bDot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
  if (aDot) {
    dot->setAttr(kAOpIdxAttr, b.getI32IntegerAttr(aDot.getOpIdx()));
    tagRole(dot.getA(), aDot);
  }
  if (bDot) {
    dot->setAttr(kBOpIdxAttr, b.getI32IntegerAttr(bDot.getOpIdx()));
    tagRole(dot.getB(), bDot);
  }
}

struct TritonWaveAMDLegalizeDotsPass
    : public impl::TritonWaveAMDLegalizeDotsBase<
          TritonWaveAMDLegalizeDotsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Architecture policy: MFMA is not supported through this path yet. Fail
    // here with a diagnostic that points at the offending op, not the bridge.
    WalkResult rejected = mod.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        auto tensorTy = dyn_cast<RankedTensorType>(result.getType());
        if (tensorTy && isMfmaEncoding(tensorTy.getEncoding())) {
          op->emitError(
              "wave_amd backend does not support AMD MFMA encodings yet");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (rejected.wasInterrupted())
      return signalPassFailure();

    // Legalize WMMA dots by attaching the typed dot plan.
    mod.walk([&](tt::DotOp dot) {
      auto resTy = dyn_cast<RankedTensorType>(dot.getType());
      if (!resTy)
        return;
      if (auto wmma =
              dyn_cast_or_null<ttg::AMDWmmaEncodingAttr>(resTy.getEncoding()))
        annotateWmmaDot(dot, wmma);
    });
  }
};

} // namespace

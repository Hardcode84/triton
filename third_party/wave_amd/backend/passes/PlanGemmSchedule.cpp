//===- PlanGemmSchedule.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wave TTGIR GEMM preparation: decide the GEMM tiling and LDS staging plan in
// TTGIR and attach it as typed attributes, so the Python bridge consumes the
// schedule mechanically instead of recomputing it.
//
//===----------------------------------------------------------------------===//

#include "wave_amd/backend/passes/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONWAVEAMDPLANGEMMSCHEDULE
#include "wave_amd/backend/passes/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

constexpr int64_t kWmmaTileDim = 16;
constexpr int64_t kSlotsPerWave = 2;

// Static GEMM tile shape (M, N, K) for a matrix-core dot, or nullopt if the
// operands are not a statically-shaped, 16-aligned f16 x f16 -> f32 matmul.
static std::optional<std::array<int64_t, 3>> gemmTileShape(tt::DotOp dot) {
  auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto resTy = dyn_cast<RankedTensorType>(dot.getType());
  if (!aTy || !bTy || !resTy)
    return std::nullopt;
  if (!isa_and_nonnull<ttg::AMDWmmaEncodingAttr>(resTy.getEncoding()))
    return std::nullopt;
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || resTy.getRank() != 2)
    return std::nullopt;
  int64_t m = aTy.getDimSize(0), k = aTy.getDimSize(1);
  int64_t bk = bTy.getDimSize(0), n = bTy.getDimSize(1);
  if (bk != k || resTy.getDimSize(0) != m || resTy.getDimSize(1) != n)
    return std::nullopt;
  for (int64_t dim : {m, n, k})
    if (dim <= 0 || dim % kWmmaTileDim != 0)
      return std::nullopt;
  return std::array<int64_t, 3>{m, n, k};
}

struct TritonWaveAMDPlanGemmSchedulePass
    : public impl::TritonWaveAMDPlanGemmScheduleBase<
          TritonWaveAMDPlanGemmSchedulePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Builder b(&getContext());

    int64_t numWarps = ttg::lookupNumWarps(mod);
    int64_t threadsPerWarp = kWmmaTileDim * 2; // sensible default
    if (auto attr = mod->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp))
      threadsPerWarp = attr.getInt();

    // LDS staging is sized per wave: two slots (A and B operand fragments),
    // each holding fragmentRegisters dwords for every lane in the wave.
    int64_t dwordsPerSlot = threadsPerWarp * fragmentRegisters;
    int64_t ldsBytes = kSlotsPerWave * dwordsPerSlot * numWarps * 4;

    mod.walk([&](tt::DotOp dot) {
      std::optional<std::array<int64_t, 3>> mnk = gemmTileShape(dot);
      if (!mnk)
        return;
      auto [m, n, k] = *mnk;
      dot->setAttr("waveamd.gemm.m_tiles",
                   b.getI32IntegerAttr(m / kWmmaTileDim));
      dot->setAttr("waveamd.gemm.n_tiles",
                   b.getI32IntegerAttr(n / kWmmaTileDim));
      dot->setAttr("waveamd.gemm.k_steps",
                   b.getI32IntegerAttr(k / kWmmaTileDim));
      dot->setAttr("waveamd.gemm.lds_slots_per_wave",
                   b.getI32IntegerAttr(kSlotsPerWave));
      dot->setAttr("waveamd.gemm.lds_dwords_per_slot",
                   b.getI32IntegerAttr(dwordsPerSlot));
      dot->setAttr("waveamd.gemm.lds_bytes", b.getI32IntegerAttr(ldsBytes));
    });
  }
};

} // namespace

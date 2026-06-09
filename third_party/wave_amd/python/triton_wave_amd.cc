#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "wave_amd/backend/passes/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

const char *getCmpIPredicateName(mlir::arith::CmpIPredicate predicate) {
  switch (predicate) {
  case mlir::arith::CmpIPredicate::eq:
    return "eq";
  case mlir::arith::CmpIPredicate::ne:
    return "ne";
  case mlir::arith::CmpIPredicate::slt:
    return "slt";
  case mlir::arith::CmpIPredicate::sle:
    return "sle";
  case mlir::arith::CmpIPredicate::sgt:
    return "sgt";
  case mlir::arith::CmpIPredicate::sge:
    return "sge";
  case mlir::arith::CmpIPredicate::ult:
    return "ult";
  case mlir::arith::CmpIPredicate::ule:
    return "ule";
  case mlir::arith::CmpIPredicate::ugt:
    return "ugt";
  case mlir::arith::CmpIPredicate::uge:
    return "uge";
  }
  return nullptr;
}

py::object getScalarTypeName(mlir::Type type) {
  if (auto integerType = llvm::dyn_cast<mlir::IntegerType>(type)) {
    switch (integerType.getWidth()) {
    case 1:
      return py::str("i1");
    case 8:
      return py::str("i8");
    case 32:
      return py::str("i32");
    case 64:
      return py::str("i64");
    }
  }
  if (type.isF16())
    return py::str("f16");
  if (type.isBF16())
    return py::str("bf16");
  if (type.isF32())
    return py::str("f32");
  return py::none();
}

py::object packSplatConstant(py::object value, mlir::Type elementType,
                             std::optional<int64_t> width) {
  py::object typeName = getScalarTypeName(elementType);
  if (typeName.is_none())
    return py::none();

  py::tuple result(3);
  result[0] = value;
  result[1] = typeName;
  result[2] = width ? py::object(py::int_(*width)) : py::object(py::none());
  return result;
}

py::object packTensorInfo(mlir::Type type) {
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor)
    return py::none();

  mlir::Type elementType = tensor.getElementType();
  bool isPointer = false;
  if (auto pointerType =
          llvm::dyn_cast<mlir::triton::PointerType>(elementType)) {
    elementType = pointerType.getPointeeType();
    isPointer = true;
  }

  py::object elementTypeName = getScalarTypeName(elementType);
  if (elementTypeName.is_none())
    return py::none();

  py::tuple shape(tensor.getRank());
  for (auto [index, dim] : llvm::enumerate(tensor.getShape()))
    shape[index] = py::int_(dim);

  py::tuple result(3);
  result[0] = shape;
  result[1] = elementTypeName;
  result[2] = py::bool_(isPointer);
  return result;
}

} // namespace

void init_triton_wave_amd(py::module &&m) {
  m.doc() = "Wave AMDGPU backend native hooks";

  // Wave-owned Triton-to-TritonGPU conversion. Mirrors the option order of the
  // base passes.ttir.add_convert_to_ttgpuir so the pipeline can swap producers.
  m.def("add_convert_to_ttgpuir",
        [](mlir::PassManager &pm, const std::string &target, int numWarps,
           int threadsPerWarp, int numCTAs) {
          mlir::TritonWaveAMDConvertToTTGPUIROptions options;
          options.target = target;
          options.numWarps = numWarps;
          options.threadsPerWarp = threadsPerWarp;
          options.numCTAs = numCTAs;
          pm.addPass(mlir::createTritonWaveAMDConvertToTTGPUIR(options));
        });

  // Wave-owned AMD matrix-core dot legalization. Mirrors the option order of
  // amd.passes.ttgpuir.add_accelerate_matmul so the pipeline can swap
  // producers.
  m.def("add_accelerate_matmul",
        [](mlir::PassManager &pm, const std::string &gfxArch,
           int matrixInstructionSize, int kPack) {
          mlir::TritonWaveAMDAccelerateMatmulOptions options;
          options.gfxArch = gfxArch;
          options.matrixInstructionSize = matrixInstructionSize;
          options.kPack = kPack;
          pm.addPass(mlir::createTritonWaveAMDAccelerateMatmul(options));
        });

  // TTGIR GEMM preparation: legalize dots against Wave AMD architecture policy
  // and attach typed dot-plan attributes for the bridge.
  m.def("add_legalize_dots", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonWaveAMDLegalizeDots());
  });

  // TTGIR GEMM preparation: decide GEMM tiling and LDS staging so the bridge
  // consumes the schedule mechanically.
  m.def("add_plan_gemm_schedule", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonWaveAMDPlanGemmSchedule());
  });

  // TTGIR GEMM preparation: decide buffer descriptor ranges for static
  // footprints so the bridge emits Wave buffer descriptors mechanically.
  m.def("add_plan_buffer_descriptors", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createTritonWaveAMDPlanBufferDescriptors());
  });

  m.def("get_program_id_axis", [](mlir::Operation *op) -> py::object {
    if (!op)
      return py::none();
    if (auto programId = llvm::dyn_cast<mlir::triton::GetProgramIdOp>(op))
      return py::int_(programId.getAxisAsInt());
    return py::none();
  });

  m.def("get_cmpi_predicate", [](mlir::Operation *op) -> py::object {
    if (!op)
      return py::none();
    if (auto cmp = llvm::dyn_cast<mlir::arith::CmpIOp>(op)) {
      if (const char *predicate = getCmpIPredicateName(cmp.getPredicate()))
        return py::str(predicate);
    }
    return py::none();
  });

  m.def("get_arith_constant_splat", [](mlir::Operation *op) -> py::object {
    if (!op)
      return py::none();
    auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(op);
    if (!constant)
      return py::none();

    mlir::Attribute value = constant.getValue();
    if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(value))
      return packSplatConstant(py::int_(integer.getValue().getSExtValue()),
                               integer.getType(), std::nullopt);
    if (auto fp = llvm::dyn_cast<mlir::FloatAttr>(value))
      return packSplatConstant(py::float_(fp.getValueAsDouble()), fp.getType(),
                               std::nullopt);

    auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(value);
    if (!dense || !dense.isSplat())
      return py::none();

    mlir::Type elementType = dense.getElementType();
    if (llvm::isa<mlir::IntegerType>(elementType)) {
      return packSplatConstant(
          py::int_(dense.getSplatValue<llvm::APInt>().getSExtValue()),
          elementType, dense.getNumElements());
    }
    if (llvm::isa<mlir::FloatType>(elementType)) {
      return packSplatConstant(
          py::float_(dense.getSplatValue<llvm::APFloat>().convertToDouble()),
          elementType, dense.getNumElements());
    }
    return py::none();
  });

  m.def("get_result_tensor_info",
        [](mlir::Operation *op, int64_t resultIndex) -> py::object {
          if (!op || resultIndex < 0 ||
              resultIndex >= static_cast<int64_t>(op->getNumResults()))
            return py::none();
          return packTensorInfo(op->getResult(resultIndex).getType());
        });
}

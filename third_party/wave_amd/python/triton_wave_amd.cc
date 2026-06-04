#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

} // namespace

void init_triton_wave_amd(py::module &&m) {
  m.doc() = "Wave AMDGPU backend native hooks";

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
}

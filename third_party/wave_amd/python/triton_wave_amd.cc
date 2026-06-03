#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Casting.h"
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
}

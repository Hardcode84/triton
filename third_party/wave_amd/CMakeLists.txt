# Triton-tree Wave AMD TTIR/TTGIR preparation passes (built against Triton's
# LLVM and registered into triton-opt / the TritonWaveAMD plugin).
add_subdirectory(backend/passes)

# Build the Wave translation tool as a Triton-managed backend artifact.
set(WAVE_AMD_WAVE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/wave)
set(WAVE_AMD_WAVE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/wave)
set(WAVE_AMD_WAVE_LLVM_INSTALL_DIR ${WAVE_AMD_WAVE_SOURCE_DIR}/build/llvm-install)
set(WAVE_AMD_WAVE_LLVM_STAMP ${WAVE_AMD_WAVE_LLVM_INSTALL_DIR}/.wave-mlir-commit)
set(WAVE_AMD_WAVE_TRANSLATE ${WAVE_AMD_WAVE_BINARY_DIR}/bin/wave-translate)
set(WAVE_AMD_WAVE_PIPELINES ${WAVE_AMD_WAVE_BINARY_DIR}/share/wave-mlir/pipelines/pipelines.mlir)

include(ProcessorCount)
ProcessorCount(WAVE_AMD_HOST_PROCESSOR_COUNT)
set(WAVE_AMD_LLVM_BUILD_JOBS "" CACHE STRING
    "Parallelism passed to Wave's build_tools/build_llvm.py")
if(NOT WAVE_AMD_LLVM_BUILD_JOBS AND WAVE_AMD_HOST_PROCESSOR_COUNT)
  set(WAVE_AMD_LLVM_BUILD_JOBS "${WAVE_AMD_HOST_PROCESSOR_COUNT}" CACHE STRING
      "Parallelism passed to Wave's build_tools/build_llvm.py" FORCE)
endif()

get_filename_component(WAVE_AMD_CMAKE_BIN_DIR "${CMAKE_COMMAND}" DIRECTORY)
get_filename_component(WAVE_AMD_NINJA_BIN_DIR "${CMAKE_MAKE_PROGRAM}" DIRECTORY)
set(WAVE_AMD_BUILD_PATH "${WAVE_AMD_CMAKE_BIN_DIR}:${WAVE_AMD_NINJA_BIN_DIR}:$ENV{PATH}")

set(WAVE_AMD_BUILD_LLVM_ARGS)
if(WAVE_AMD_LLVM_BUILD_JOBS)
  list(APPEND WAVE_AMD_BUILD_LLVM_ARGS --jobs ${WAVE_AMD_LLVM_BUILD_JOBS})
endif()

add_custom_command(
  OUTPUT ${WAVE_AMD_WAVE_LLVM_STAMP}
  COMMAND ${CMAKE_COMMAND} -E env
          --unset=LLVM_INSTALL_DIR
          --unset=LLVM_PROJECT_SOURCE_DIR
          --unset=LLVM_COMMIT
          "PATH=${WAVE_AMD_BUILD_PATH}"
          ${Python3_EXECUTABLE}
          ${WAVE_AMD_WAVE_SOURCE_DIR}/build_tools/build_llvm.py
          --python-bindings
          ${WAVE_AMD_BUILD_LLVM_ARGS}
  DEPENDS ${WAVE_AMD_WAVE_SOURCE_DIR}/llvm-commit.txt
          ${WAVE_AMD_WAVE_SOURCE_DIR}/build_tools/build_llvm.py
  WORKING_DIRECTORY ${WAVE_AMD_WAVE_SOURCE_DIR}
  COMMENT "Building Wave-pinned LLVM/MLIR from llvm-commit.txt"
  VERBATIM)

add_custom_target(WaveAMDLLVM DEPENDS ${WAVE_AMD_WAVE_LLVM_STAMP})

add_custom_command(
  OUTPUT ${WAVE_AMD_WAVE_BINARY_DIR}/CMakeCache.txt
  COMMAND ${CMAKE_COMMAND} -E env
          "PATH=${WAVE_AMD_BUILD_PATH}"
          ${CMAKE_COMMAND}
          -S ${WAVE_AMD_WAVE_SOURCE_DIR}
          -B ${WAVE_AMD_WAVE_BINARY_DIR}
          -G Ninja
          -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
          -DLLVM_INSTALL_DIR=${WAVE_AMD_WAVE_LLVM_INSTALL_DIR}
          -DWAVE_ENABLE_PYTHON_BINDINGS=ON
          -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
  DEPENDS WaveAMDLLVM
  COMMENT "Configuring Wave against its pinned LLVM/MLIR build"
  VERBATIM)

add_custom_command(
  OUTPUT ${WAVE_AMD_WAVE_TRANSLATE}
  COMMAND ${CMAKE_COMMAND} -E env
          "PATH=${WAVE_AMD_BUILD_PATH}"
          ${CMAKE_COMMAND}
          --build ${WAVE_AMD_WAVE_BINARY_DIR}
          --target wave-translate WavePythonModules
  DEPENDS ${WAVE_AMD_WAVE_BINARY_DIR}/CMakeCache.txt
  COMMENT "Building Wave wave-translate and Python bindings"
  VERBATIM)

add_custom_target(WaveAMDWaveTranslate
  DEPENDS ${WAVE_AMD_WAVE_TRANSLATE})

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonWaveAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_wave_amd.cc)
  target_link_libraries(TritonWaveAMD PRIVATE Python3::Module pybind11::headers
                        TritonWaveAMDBackendPasses)
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  if(NOT DEFINED TRITON_WHEEL_DIR)
    message(FATAL_ERROR "TRITON_WHEEL_DIR must be set to package wave-translate.")
  endif()
  set(WAVE_AMD_BACKEND_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backend/bin")
  set(WAVE_AMD_BACKEND_PIPELINES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/backend/share/wave-mlir/pipelines")
  set(WAVE_AMD_PACKAGED_WAVE_TRANSLATE "${WAVE_AMD_BACKEND_BIN_DIR}/wave-translate")
  set(WAVE_AMD_PACKAGED_PIPELINES "${WAVE_AMD_BACKEND_PIPELINES_DIR}/pipelines.mlir")
  add_custom_command(
    OUTPUT "${WAVE_AMD_PACKAGED_WAVE_TRANSLATE}"
           "${WAVE_AMD_PACKAGED_PIPELINES}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${WAVE_AMD_BACKEND_BIN_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${WAVE_AMD_BACKEND_PIPELINES_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WAVE_AMD_WAVE_TRANSLATE}"
            "${WAVE_AMD_PACKAGED_WAVE_TRANSLATE}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WAVE_AMD_WAVE_PIPELINES}"
            "${WAVE_AMD_PACKAGED_PIPELINES}"
    DEPENDS WaveAMDWaveTranslate
            "${WAVE_AMD_WAVE_PIPELINES}"
    COMMENT "Packaging wave-translate for the wave_amd backend"
    VERBATIM)
  add_custom_target(WaveAMDPackagedWaveTranslate ALL
    DEPENDS "${WAVE_AMD_PACKAGED_WAVE_TRANSLATE}"
            "${WAVE_AMD_PACKAGED_PIPELINES}")
  add_dependencies(TritonWaveAMD WaveAMDPackagedWaveTranslate)
endif()

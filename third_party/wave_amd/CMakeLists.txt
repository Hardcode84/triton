# Build the Wave translation tool as a Triton-managed backend artifact.
set(WAVE_AMD_WAVE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/wave)
set(WAVE_AMD_WAVE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/wave)

include_directories(${WAVE_AMD_WAVE_SOURCE_DIR}/include)
include_directories(${WAVE_AMD_WAVE_BINARY_DIR}/include)
include_directories(${WAVE_AMD_WAVE_SOURCE_DIR}/third_party/ixsimpl/include)

if(NOT DEFINED LLVM_MAIN_SRC_DIR)
  set(LLVM_MAIN_SRC_DIR "${WAVE_AMD_WAVE_SOURCE_DIR}/build/_deps/llvm-project/llvm")
endif()
if(NOT DEFINED LLVM_BINARY_DIR)
  set(LLVM_BINARY_DIR "${WAVE_AMD_WAVE_SOURCE_DIR}/build/llvm-build")
endif()

if(NOT TARGET wave-translate)
  add_subdirectory(${WAVE_AMD_WAVE_SOURCE_DIR}/third_party ${WAVE_AMD_WAVE_BINARY_DIR}/third_party)
  add_subdirectory(${WAVE_AMD_WAVE_SOURCE_DIR}/include ${WAVE_AMD_WAVE_BINARY_DIR}/include)
  add_subdirectory(${WAVE_AMD_WAVE_SOURCE_DIR}/lib ${WAVE_AMD_WAVE_BINARY_DIR}/lib)
  add_subdirectory(${WAVE_AMD_WAVE_SOURCE_DIR}/tools/wave-translate ${WAVE_AMD_WAVE_BINARY_DIR}/tools/wave-translate)
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonWaveAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_wave_amd.cc)
  target_link_libraries(TritonWaveAMD PRIVATE Python3::Module pybind11::headers)

  if(NOT DEFINED TRITON_WHEEL_DIR)
    message(FATAL_ERROR "TRITON_WHEEL_DIR must be set to package wave-translate.")
  endif()
  set(WAVE_AMD_BACKEND_BIN_DIR "${TRITON_WHEEL_DIR}/backends/wave_amd/bin")
  set(WAVE_AMD_PACKAGED_WAVE_TRANSLATE "${WAVE_AMD_BACKEND_BIN_DIR}/wave-translate")
  add_custom_command(
    OUTPUT "${WAVE_AMD_PACKAGED_WAVE_TRANSLATE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${WAVE_AMD_BACKEND_BIN_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:wave-translate>"
            "${WAVE_AMD_PACKAGED_WAVE_TRANSLATE}"
    DEPENDS wave-translate
    COMMENT "Packaging wave-translate for the wave_amd backend"
    VERBATIM)
  add_custom_target(WaveAMDPackagedWaveTranslate ALL
    DEPENDS "${WAVE_AMD_PACKAGED_WAVE_TRANSLATE}")
  add_dependencies(TritonWaveAMD WaveAMDPackagedWaveTranslate)
endif()

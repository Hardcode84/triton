# M0 only exposes an empty native backend module so libtriton can initialize the
# in-tree backend. Wave native libraries are linked in a later slice.
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonWaveAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_wave_amd.cc)
  target_link_libraries(TritonWaveAMD PRIVATE Python3::Module pybind11::headers)
endif()

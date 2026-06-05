cmake_minimum_required(VERSION 3.18)
project(ltkit_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only engine test — no libtorch needed, proves the core in isolation.
add_executable(core_smoke tests/smoke.cpp)
target_include_directories(core_smoke PRIVATE include)

# libtorch backend test. We deliberately avoid find_package(Torch): the pip
# wheel is CUDA-enabled and drags CMake into broken CUDA-compiler detection.
# CPU-only linking against libtorch_cpu/libc10 sidesteps that. Configure with:
#   cmake -DTORCH_DIR=$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))') ..
if(DEFINED TORCH_DIR)
    add_executable(torch_smoke tests/torch_smoke.cpp)
    target_include_directories(torch_smoke PRIVATE
        include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include)
    # The wheel reports _GLIBCXX_USE_CXX11_ABI=False -> must match it.
    target_compile_definitions(torch_smoke PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(torch_smoke
        ${TORCH_DIR}/lib/libtorch.so
        ${TORCH_DIR}/lib/libtorch_cpu.so
        ${TORCH_DIR}/lib/libc10.so)
    set_target_properties(torch_smoke PROPERTIES
        BUILD_RPATH ${TORCH_DIR}/lib
        INSTALL_RPATH ${TORCH_DIR}/lib)
else()
    message(STATUS "TORCH_DIR not set — skipping torch_smoke (core_smoke only)")
endif()

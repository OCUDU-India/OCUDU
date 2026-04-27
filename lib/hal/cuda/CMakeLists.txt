# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI

if (ENABLE_DFT_GPU)
    find_package(mathdx 26.03 CONFIG QUIET COMPONENTS cufftdx
                 PATHS /opt/nvidia-mathdx-26.03.0-cuda13/nvidia/mathdx/26.03/lib/cmake)
    if (NOT mathdx_FOUND)
        find_package(cufftdx CONFIG QUIET
                     PATHS /opt/nvidia-mathdx-26.03.0-cuda13/nvidia/mathdx/26.03/lib/cmake)
    endif ()
    if (NOT TARGET cufftdx::cufftdx AND NOT TARGET mathdx::cufftdx)
        message(FATAL_ERROR "ENABLE_DFT_GPU=ON requires NVIDIA MathDx with cuFFTDx.")
    endif ()
    if (TARGET mathdx::cufftdx)
        set(CUFFTDX_TARGET mathdx::cufftdx)
    else ()
        set(CUFFTDX_TARGET cufftdx::cufftdx)
    endif ()

    add_library(ocudu_hal_dft_gpu STATIC
        prach_detector_gpu_kernel.cu
        prach_detector_gpudx_kernel.cu
    )

    set_target_properties(ocudu_hal_dft_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(ocudu_hal_dft_gpu PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/lib
    )

    target_link_libraries(ocudu_hal_dft_gpu
        PUBLIC  CUDA::cudart CUDA::cufft
        PRIVATE ${CUFFTDX_TARGET}
    )

    add_to_exported_libs(ocudu_hal_dft_gpu)
endif ()

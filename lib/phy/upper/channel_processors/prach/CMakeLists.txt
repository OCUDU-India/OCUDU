# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
# Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.


set(PRACH_DETECTOR_SOURCES
        prach_detector_generic_impl.cpp
        prach_detector_generic_thresholds.cpp
        prach_generator_impl.cpp
        prach_detector_phy_validator_impl.cpp
        factories.cpp)

if (ENABLE_DFT_GPU)
    list(APPEND PRACH_DETECTOR_SOURCES prach_detector_gpu_impl.cpp)
endif ()

add_library(ocudu_prach_detector STATIC ${PRACH_DETECTOR_SOURCES})
target_link_libraries(ocudu_prach_detector PUBLIC ocudu_ran ocuduvec)

if (ENABLE_DFT_GPU)
    target_compile_definitions(ocudu_prach_detector PRIVATE HAVE_DFT_GPU)
    target_link_libraries(ocudu_prach_detector
        PUBLIC ocudu_hal_dft_gpu
        PRIVATE CUDA::cudart CUDA::cufft)
endif ()

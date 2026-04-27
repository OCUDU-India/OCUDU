# SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI


if (DPDK_FOUND)
    add_subdirectory(dpdk)
    add_subdirectory(phy)
endif ()

if (ENABLE_DFT_GPU)
    add_subdirectory(cuda)
endif ()

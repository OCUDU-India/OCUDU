# Copyright 2025-2026 coRAN LABS Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Precompiled xSM shared library. Defines the imported 'xsm' target used by
# ocudu_xsm_context and fapi_xsm_transport. No source is compiled here.

add_library(xsm SHARED IMPORTED GLOBAL)
set_target_properties(xsm PROPERTIES
    IMPORTED_LOCATION             "${CMAKE_CURRENT_SOURCE_DIR}/libxsm.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

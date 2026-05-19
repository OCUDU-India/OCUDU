// Copyright 2025-2026 coRAN LABS Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ocudu::fapi_split_trace {

void init(bool enabled, const std::string& path, const char* binary_tag);

bool is_enabled();

void shutdown();

void event(const char* component, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

void message(const char* component,
             const char* type,
             const char* direction,
             uint16_t    sfn,
             uint16_t    slot,
             size_t      total_size,
             const void* data,
             size_t      data_len);

void payload_full(const char* component,
                  const char* tag,
                  uint16_t    sfn,
                  uint16_t    slot,
                  uint16_t    rnti,
                  const void* data,
                  size_t      data_len);

} // namespace ocudu::fapi_split_trace

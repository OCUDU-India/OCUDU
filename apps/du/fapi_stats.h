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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ocudu::fapi_stats {

constexpr size_t MAX_MESSAGE_STATS       = 100000;
constexpr size_t MAX_MESSAGE_TYPE_LEN    = 64;
constexpr size_t MAX_DIRECTION_LEN       = 16;
constexpr size_t MAX_MESSAGE_CONTENT_LEN = 8192;


struct message_stat {
  uint64_t timestamp_ns                             = 0;
  uint64_t ipc_latency_ns                           = 0;
  char     message_type[MAX_MESSAGE_TYPE_LEN]       = {};
  char     direction[MAX_DIRECTION_LEN]             = {};
  int32_t  sfn                                      = 0;
  int32_t  slot                                     = 0;
  int32_t  pdu_size                                 = 0;
  int32_t  num_pdus                                 = 0;
  char     message_content[MAX_MESSAGE_CONTENT_LEN] = {};
};


void initialize(bool enabled, const std::string& output_path, bool add_timestamp = false);

bool is_enabled();


void record(const char* msg_type,
            const char* direction,
            int         sfn,
            int         slot,
            int         pdu_size,
            int         num_pdus,
            uint64_t    ipc_latency_ns,
            const char* content);

void dump_to_json();

void shutdown();

uint64_t timestamp_ns();

} // namespace ocudu::fapi_stats

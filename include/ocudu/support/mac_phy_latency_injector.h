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

#include <cstdint>

namespace ocudu::mac_phy_latency_injector {

enum class msg_kind : uint8_t {
  DL_TTI_REQUEST  = 0,
  UL_TTI_REQUEST  = 1,
  UL_DCI_REQUEST  = 2,
  TX_DATA_REQUEST = 3
};

void initialize_from_env();

void initialize(bool     enabled,
                uint64_t dl_tti_mean_ns,
                uint64_t ul_tti_mean_ns,
                uint64_t ul_dci_mean_ns,
                uint64_t tx_data_mean_ns,
                uint32_t jitter_pct);

bool is_enabled();

void inject(msg_kind kind);

void shutdown();

} // namespace ocudu::mac_phy_latency_injector

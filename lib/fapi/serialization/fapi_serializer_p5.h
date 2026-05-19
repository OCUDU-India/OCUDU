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

#include "fapi_serializer_common.h"
#include "ocudu/fapi/cell_config.h"
#include "ocudu/fapi/p5/p5_messages.h"

namespace ocudu {
namespace fapi_serial {

void serialize(buffer_writer& w, const fapi::param_request& msg);
void deserialize(buffer_reader& r, fapi::param_request& msg);

void serialize(buffer_writer& w, const fapi::param_response& msg);
void deserialize(buffer_reader& r, fapi::param_response& msg);

void serialize(buffer_writer& w, const fapi::config_response& msg);
void deserialize(buffer_reader& r, fapi::config_response& msg);

void serialize(buffer_writer& w, const fapi::start_request& msg);
void deserialize(buffer_reader& r, fapi::start_request& msg);

void serialize(buffer_writer& w, const fapi::stop_request& msg);
void deserialize(buffer_reader& r, fapi::stop_request& msg);

void serialize(buffer_writer& w, const fapi::stop_indication& msg);
void deserialize(buffer_reader& r, fapi::stop_indication& msg);

void serialize(buffer_writer& w, const fapi::carrier_config& cfg);
void deserialize(buffer_reader& r, fapi::carrier_config& cfg);

void serialize(buffer_writer& w, const rach_config_common& cfg);
void deserialize(buffer_reader& r, rach_config_common& cfg);

void serialize(buffer_writer& w, const ssb_configuration& cfg);
void deserialize(buffer_reader& r, ssb_configuration& cfg);

void serialize(buffer_writer& w, const fapi::cell_configuration& cfg);
void deserialize(buffer_reader& r, fapi::cell_configuration& cfg);

void serialize(buffer_writer& w, const fapi::config_request& msg);
void deserialize(buffer_reader& r, fapi::config_request& msg);

} // namespace fapi_serial
} // namespace ocudu

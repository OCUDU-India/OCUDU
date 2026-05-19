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
#include "ocudu/fapi/p7/messages/ul_tti_request.h"

namespace ocudu {
namespace fapi_serial {

void serialize(buffer_writer& w, const fapi::ul_prach_pdu& pdu);
void deserialize(buffer_reader& r, fapi::ul_prach_pdu& pdu);

void serialize(buffer_writer& w, const fapi::ul_pusch_pdu& pdu);
void deserialize(buffer_reader& r, fapi::ul_pusch_pdu& pdu);

void serialize(buffer_writer& w, const fapi::ul_pucch_pdu& pdu);
void deserialize(buffer_reader& r, fapi::ul_pucch_pdu& pdu);

void serialize(buffer_writer& w, const fapi::ul_srs_pdu& pdu);
void deserialize(buffer_reader& r, fapi::ul_srs_pdu& pdu);

void serialize(buffer_writer& w, const fapi::ul_tti_request_pdu& pdu);
void deserialize(buffer_reader& r, fapi::ul_tti_request_pdu& pdu);

void serialize(buffer_writer& w, const fapi::ul_tti_request& msg);
void deserialize(buffer_reader& r, fapi::ul_tti_request& msg);

} // namespace fapi_serial
} // namespace ocudu

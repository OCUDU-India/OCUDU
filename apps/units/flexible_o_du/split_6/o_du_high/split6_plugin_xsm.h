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

#include "split6_plugin.h"
#include <memory>

namespace ocudu {

class split6_plugin_xsm : public split6_plugin
{
public:
  explicit split6_plugin_xsm(std::string_view app_name) {}

  void on_parsing_configuration_registration(CLI::App& app) override {}

  bool on_configuration_validation() const override { return true; }

  void on_loggers_registration() override {}

  void fill_worker_manager_config(worker_manager_config& config) override {}

  std::unique_ptr<fapi_adaptor::phy_fapi_adaptor>
  create_fapi_adaptor(const odu::du_high_configuration& du_high_cfg,
                      const o_du_unit_dependencies&     dependencies) override;
};

} // namespace ocudu

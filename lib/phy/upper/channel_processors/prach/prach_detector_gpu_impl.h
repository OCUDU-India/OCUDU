// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/phy/upper/channel_processors/prach/prach_detector.h"
#include "ocudu/phy/upper/channel_processors/prach/prach_generator.h"
#include "ocudu/ran/prach/prach_format_type.h"
#include "ocudu/ran/prach/restricted_set_config.h"
#include <cstdint>
#include <memory>

namespace ocudu {

class prach_buffer;

class prach_detector_gpu_impl : public prach_detector
{
public:
  prach_detector_gpu_impl(std::unique_ptr<prach_generator> generator_,
                          std::unique_ptr<prach_detector>  fallback_,
                          unsigned                         idft_long_size_,
                          unsigned                         idft_short_size_);

  ~prach_detector_gpu_impl() override;

  prach_detection_result detect(const prach_buffer& input, const configuration& config) override;

private:
  void ensure_resources();
  void release_resources();

  std::unique_ptr<prach_generator> generator;
  std::unique_ptr<prach_detector>  fallback;

  unsigned idft_long_size  = 0;
  unsigned idft_short_size = 0;

  void* stream         = nullptr;
  bool  resources_init = false;
  void* plan_cache     = nullptr;
  void* graph_cache    = nullptr;

  static constexpr unsigned MAX_BATCH = 64u * 12u;

  void* d_root            = nullptr;
  void* d_combined        = nullptr;
  void* d_preamble        = nullptr;
  void* d_idft            = nullptr;
  void* d_mod_sq          = nullptr;
  void* d_mod_sq_combined = nullptr;
  void* d_num             = nullptr;
  void* d_den             = nullptr;
  void* d_argmax_idx      = nullptr;
  void* d_argmax_val      = nullptr;
  void* d_num_at_argmax   = nullptr;
  void* d_window_starts   = nullptr;

  void* h_root          = nullptr;
  void* h_preamble      = nullptr;
  void* h_argmax_idx    = nullptr;
  void* h_argmax_val    = nullptr;
  void* h_num_at_argmax = nullptr;
  void* h_window_starts = nullptr;

  bool                  roots_cached               = false;
  prach_format_type     cached_root_format         = prach_format_type::invalid;
  unsigned              cached_root_index          = 0;
  restricted_set_config cached_root_restricted_set = restricted_set_config::UNRESTRICTED;
  unsigned              cached_root_zcz            = 0;
  unsigned              cached_root_nof_sequences  = 0;
  unsigned              cached_root_nof_shifts     = 0;
  unsigned              cached_root_L_ra           = 0;

  unsigned long long total_detects      = 0;
  unsigned long long total_graph_builds = 0;
  unsigned long long sum_detect_ns      = 0;
  unsigned long long max_detect_ns      = 0;
  unsigned long long min_detect_ns      = 0;
};

std::unique_ptr<prach_detector> create_prach_detector_gpu(std::unique_ptr<prach_generator> generator,
                                                          std::unique_ptr<prach_detector>  fallback,
                                                          unsigned                         idft_long_size,
                                                          unsigned                         idft_short_size);

} // namespace ocudu

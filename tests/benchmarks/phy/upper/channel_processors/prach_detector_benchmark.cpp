// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/upper/channel_processors/prach/factories.h"
#include "ocudu/phy/upper/channel_processors/prach/formatters.h"
#include "ocudu/support/benchmark_utils.h"
#include "ocudu/support/error_handling.h"
#include "ocudu/support/math/complex_normal_random.h"
#include <getopt.h>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ocudu;

static uint64_t                    nof_repetitions   = 1000;
static std::set<unsigned>          set_nof_rx_ports  = {1, 2, 4};
static std::set<prach_format_type> set_format        = {prach_format_type::zero, prach_format_type::B4};
static std::set<unsigned>          set_zcz           = {0, 1, 14};
static std::set<unsigned>          set_nof_preambles = {4, 64};

static std::string backend_choice    = "fftw";
static bool        validate_gpu_full = false;
static unsigned    warmup_runs       = 8;

static std::mt19937 rgen(0);

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-R repetitions] [-b backend]\n", prog);
  fmt::print("\t-R Repetitions [Default {}]\n", nof_repetitions);
  fmt::print("\t-p Fix the number of ports\n");
  fmt::print("\t-f Fix the PRACH format\n");
  fmt::print("\t-z Fix the Zero Correlation Zone configuration index.\n");
  fmt::print("\t-b Backend: fftw | gpu_full | all [Default {}]\n", backend_choice);
  fmt::print("\t-v Validate gpu_full numerical equivalence against fftw and exit\n");
  fmt::print("\t-W Warmup runs before each measurement (0 measures cold-start) [Default {}]\n", warmup_runs);
  fmt::print("\t-h Show this message\n");
}

static int parse_args(int argc, char** argv)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "R:p:f:z:b:W:vh")) != -1) {
    switch (opt) {
      case 'R':
        nof_repetitions = std::strtol(optarg, nullptr, 10);
        break;
      case 'p':
        set_nof_rx_ports = {static_cast<unsigned>(std::strtol(optarg, nullptr, 10))};
        break;
      case 'f':
        set_format = {to_prach_format_type(optarg)};
        break;
      case 'z':
        set_zcz = {static_cast<unsigned>(std::strtol(optarg, nullptr, 10))};
        break;
      case 'b':
        backend_choice = optarg;
        break;
      case 'v':
        validate_gpu_full = true;
        break;
      case 'W':
        warmup_runs = static_cast<unsigned>(std::strtol(optarg, nullptr, 10));
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(0);
    }
  }
  return 0;
}

static std::shared_ptr<dft_processor_factory> make_dft_factory(const std::string& name)
{
  if (name == "fftw" || name == "gpu_full") {
    return create_dft_processor_factory_fftw_fast();
  }
  return nullptr;
}

static std::shared_ptr<prach_detector_factory> make_detector_factory(const std::string&                     backend,
                                                                     std::shared_ptr<dft_processor_factory> dft_factory)
{
  std::shared_ptr<prach_generator_factory> prach_gen_factory = create_prach_generator_factory_sw();
  report_fatal_error_if_not(prach_gen_factory, "Failed to create PRACH generator factory.");
  if (backend == "gpu_full") {
    return create_prach_detector_factory_gpu(std::move(dft_factory), prach_gen_factory);
  }
  return create_prach_detector_factory_sw(std::move(dft_factory), prach_gen_factory);
}

static std::unique_ptr<prach_buffer> create_buffer(prach_format_type format, unsigned nof_antennas)
{
  complex_normal_distribution<cf_t> complex_float_dist(cf_t(1, 1), std::sqrt(2.0F));

  std::unique_ptr<prach_buffer> buffer;

  if (is_long_preamble(format)) {
    buffer = create_prach_buffer_long(nof_antennas, 1);
  } else {
    buffer = create_prach_buffer_short(nof_antennas, 1, 1);
  }
  report_fatal_error_if_not(buffer, "Failed to create buffer.");

  for (unsigned i_antenna = 0; i_antenna != nof_antennas; ++i_antenna) {
    for (unsigned i_symbol = 0, i_symbol_end = buffer->get_max_nof_symbols(); i_symbol != i_symbol_end; ++i_symbol) {
      span<cbf16_t> prach_symbol = buffer->get_symbol(i_antenna, 0, 0, i_symbol);
      for (cbf16_t& sample : prach_symbol) {
        sample = complex_float_dist(rgen);
      }
    }
  }

  return buffer;
}

std::vector<prach_detector::configuration> generate_test_cases()
{
  std::vector<prach_detector::configuration> test_cases;
  for (unsigned nof_rx_ports : set_nof_rx_ports) {
    for (prach_format_type format : set_format) {
      std::uniform_int_distribution<unsigned> root_sequence_index_dist(0, (is_long_preamble(format) ? 837 : 138));

      for (unsigned zcz : set_zcz) {
        if (((format == prach_format_type::zero) && (zcz == 14)) || ((format == prach_format_type::B4) && (zcz == 1))) {
          continue;
        }

        for (unsigned nof_preambles : set_nof_preambles) {
          std::uniform_int_distribution<unsigned> start_preamble_index_dist(0, 64 - nof_preambles);

          test_cases.emplace_back();
          prach_detector::configuration& config = test_cases.back();

          prach_subcarrier_spacing ra_scs = prach_subcarrier_spacing::kHz30;
          if (is_long_preamble(format)) {
            if (format == prach_format_type::three) {
              ra_scs = prach_subcarrier_spacing::kHz5;
            } else {
              ra_scs = prach_subcarrier_spacing::kHz1_25;
            }
          }

          config.root_sequence_index   = root_sequence_index_dist(rgen);
          config.format                = format;
          config.restricted_set        = restricted_set_config::UNRESTRICTED;
          config.zero_correlation_zone = zcz;
          config.start_preamble_index  = start_preamble_index_dist(rgen);
          config.nof_preamble_indices  = nof_preambles;
          config.ra_scs                = ra_scs;
          config.nof_rx_ports          = nof_rx_ports;
        }
      }
    }
  }

  return test_cases;
}

static void benchmark_backend(const std::string&                                backend_name,
                              prach_detector_factory&                           factory,
                              const std::vector<prach_detector::configuration>& configurations,
                              prach_detector_validator&                         validator)
{
  benchmarker                     perf_meas(fmt::format("PRACH detector [{}]", backend_name), nof_repetitions);
  std::unique_ptr<prach_detector> detector = factory.create();
  report_fatal_error_if_not(detector, "Failed to create PRACH detector for backend '{}'.", backend_name);

  for (const prach_detector::configuration& config : configurations) {
    report_fatal_error_if_not(validator.is_valid(config), "Invalid PRACH detector configuration {}.", config);

    std::unique_ptr<prach_buffer> buffer = create_buffer(config.format, config.nof_rx_ports);
    report_fatal_error_if_not(buffer, "Failed to create buffer.");

    fmt::memory_buffer meas_description;
    fmt::format_to(std::back_inserter(meas_description), "{}", config);

    for (unsigned w = 0; w < warmup_runs; ++w) {
      detector->detect(*buffer, config);
    }

    perf_meas.new_measure(
        to_string(meas_description), 1, [&detector, &buffer, &config]() { detector->detect(*buffer, config); });
  }

  perf_meas.print_percentiles_time("microseconds", 1e-3);
}

static int run_validation()
{
  std::vector<prach_detector::configuration> configurations = generate_test_cases();

  std::shared_ptr<dft_processor_factory>  fftw_dft = make_dft_factory("fftw");
  std::shared_ptr<dft_processor_factory>  gpu_dft  = make_dft_factory("gpu_full");
  std::shared_ptr<prach_detector_factory> fftw_det = make_detector_factory("fftw", fftw_dft);
  std::shared_ptr<prach_detector_factory> gpu_det  = make_detector_factory("gpu_full", gpu_dft);
  if (!fftw_det || !gpu_det) {
    fmt::print("Validation skipped: one of the detector factories is unavailable.\n");
    return 0;
  }
  std::unique_ptr<prach_detector>           fftw_d    = fftw_det->create();
  std::unique_ptr<prach_detector>           gpu_d     = gpu_det->create();
  std::unique_ptr<prach_detector_validator> validator = fftw_det->create_validator();

  unsigned diverged = 0;
  for (const prach_detector::configuration& config : configurations) {
    if (!validator->is_valid(config)) {
      continue;
    }
    std::unique_ptr<prach_buffer> buffer = create_buffer(config.format, config.nof_rx_ports);
    prach_detection_result        r_cpu  = fftw_d->detect(*buffer, config);
    prach_detection_result        r_gpu  = gpu_d->detect(*buffer, config);

    bool ok = true;
    if (std::abs(r_cpu.rssi_dB - r_gpu.rssi_dB) > 1e-3F) {
      ok = false;
    }
    if (r_cpu.preambles.size() != r_gpu.preambles.size()) {
      ok = false;
    } else {
      for (size_t i = 0; i < r_cpu.preambles.size(); ++i) {
        const auto& a = r_cpu.preambles[i];
        const auto& b = r_gpu.preambles[i];
        if (a.preamble_index != b.preamble_index ||
            std::abs(a.detection_metric - b.detection_metric) > 1e-3F * std::abs(a.detection_metric) + 1e-4F ||
            std::abs(a.preamble_power_dB - b.preamble_power_dB) > 0.1F) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) {
      ++diverged;
      fmt::print("DIVERGENCE on {}: cpu rssi={} preambles={}, gpu rssi={} preambles={}\n",
                 config,
                 r_cpu.rssi_dB,
                 r_cpu.preambles.size(),
                 r_gpu.rssi_dB,
                 r_gpu.preambles.size());
    }
  }

  if (diverged == 0) {
    fmt::print("Validation OK: fftw and gpu_full produce equivalent results across {} test configs.\n",
               configurations.size());
    return 0;
  }
  fmt::print("Validation FAILED: {} configs diverged.\n", diverged);
  return 1;
}

int main(int argc, char** argv)
{
  int ret = parse_args(argc, argv);
  if (ret < 0) {
    return ret;
  }

  if (validate_gpu_full) {
    return run_validation();
  }

  std::vector<std::string> backends;
  if (backend_choice == "all") {
    backends = {"fftw", "gpu_full"};
  } else {
    backends = {backend_choice};
  }

  std::vector<prach_detector::configuration> configurations = generate_test_cases();

  for (const std::string& backend_name : backends) {
    std::shared_ptr<dft_processor_factory> dft_factory = make_dft_factory(backend_name);
    if (!dft_factory) {
      fmt::print("Skipping backend '{}': factory unavailable.\n", backend_name);
      continue;
    }

    std::shared_ptr<prach_detector_factory> det_factory = make_detector_factory(backend_name, dft_factory);
    if (!det_factory) {
      fmt::print("Skipping backend '{}': PRACH detector factory unavailable (e.g. no CUDA device).\n", backend_name);
      continue;
    }

    std::unique_ptr<prach_detector_validator> validator = det_factory->create_validator();
    benchmark_backend(backend_name, *det_factory, configurations, *validator);
  }

  return 0;
}

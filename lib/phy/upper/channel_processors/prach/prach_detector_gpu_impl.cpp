// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "prach_detector_gpu_impl.h"
#include "prach_detector_generic_thresholds.h"
#include "ocudu/adt/interval.h"
#include "ocudu/hal/cuda/prach_detector_gpu_kernel.h"
#include "ocudu/ocuduvec/dot_prod.h"
#include "ocudu/phy/support/prach_buffer.h"
#include "ocudu/phy/upper/channel_processors/prach/prach_detection_result.h"
#include "ocudu/ran/prach/prach_constants.h"
#include "ocudu/ran/prach/prach_cyclic_shifts.h"
#include "ocudu/ran/prach/prach_preamble_information.h"
#include "ocudu/support/error_handling.h"
#include "ocudu/support/math/math_utils.h"

#include "fmt/format.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace {

void log_gpu_info_once()
{
  static std::once_flag once;
  std::call_once(once, []() {
    int driver_version  = 0;
    int runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      fmt::print(stderr, "[prach_detector_gpu] no CUDA devices visible\n");
      return;
    }

    fmt::print(stderr,
               "[prach_detector_gpu] +======================================================================+\n"
               "[prach_detector_gpu] | GPU configuration                                                     \n"
               "[prach_detector_gpu] +------------------------------------------------------------------------+\n"
               "[prach_detector_gpu] | CUDA driver  : {}.{}\n"
               "[prach_detector_gpu] | CUDA runtime : {}.{}\n"
               "[prach_detector_gpu] | Devices visible: {}\n"
               "[prach_detector_gpu] +------------------------------------------------------------------------+\n",
               driver_version / 1000,
               (driver_version % 1000) / 10,
               runtime_version / 1000,
               (runtime_version % 1000) / 10,
               device_count);

    int active_device = 0;
    cudaGetDevice(&active_device);

    for (int dev = 0; dev < device_count; ++dev) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        fmt::print(stderr, "[prach_detector_gpu] | dev {} : cudaGetDeviceProperties failed\n", dev);
        continue;
      }

      size_t free_bytes  = 0;
      size_t total_bytes = 0;
      int    saved       = active_device;
      cudaSetDevice(dev);
      cudaMemGetInfo(&free_bytes, &total_bytes);
      cudaSetDevice(saved);

      char pci_bus_id[32] = {0};
      cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), dev);

      int gpu_clock_khz    = 0;
      int mem_clock_khz    = 0;
      int compute_mode_int = 0;
      cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev);
      cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
      cudaDeviceGetAttribute(&compute_mode_int, cudaDevAttrComputeMode, dev);

      const char* compute_mode_str = "default";
      switch (compute_mode_int) {
        case cudaComputeModeExclusive:
          compute_mode_str = "exclusive";
          break;
        case cudaComputeModeProhibited:
          compute_mode_str = "prohibited";
          break;
        case cudaComputeModeExclusiveProcess:
          compute_mode_str = "exclusive-process";
          break;
        default:
          compute_mode_str = "default";
          break;
      }

      fmt::print(stderr,
                 "[prach_detector_gpu] | Device {}{}\n"
                 "[prach_detector_gpu] |   Name                : {}\n"
                 "[prach_detector_gpu] |   PCI bus id          : {}\n"
                 "[prach_detector_gpu] |   Compute capability  : {}.{}\n"
                 "[prach_detector_gpu] |   Compute mode        : {}\n"
                 "[prach_detector_gpu] |   Multiprocessors     : {} (warp size {})\n"
                 "[prach_detector_gpu] |   Max threads / block : {}\n"
                 "[prach_detector_gpu] |   Max threads / SM    : {}\n"
                 "[prach_detector_gpu] |   GPU clock           : {} MHz\n"
                 "[prach_detector_gpu] |   Memory clock        : {} MHz\n"
                 "[prach_detector_gpu] |   Memory bus width    : {}-bit\n"
                 "[prach_detector_gpu] |   L2 cache size       : {} KiB\n"
                 "[prach_detector_gpu] |   Total global memory : {} MiB\n"
                 "[prach_detector_gpu] |   Free  global memory : {} MiB\n"
                 "[prach_detector_gpu] |   Shared mem / block  : {} KiB\n"
                 "[prach_detector_gpu] |   ECC enabled         : {}\n"
                 "[prach_detector_gpu] |   Async engine count  : {}\n"
                 "[prach_detector_gpu] |   Concurrent kernels  : {}\n"
                 "[prach_detector_gpu] |   Unified addressing  : {}\n"
                 "[prach_detector_gpu] |   Integrated GPU      : {}\n",
                 dev,
                 (dev == active_device) ? " (active)" : "",
                 prop.name,
                 pci_bus_id,
                 prop.major,
                 prop.minor,
                 compute_mode_str,
                 prop.multiProcessorCount,
                 prop.warpSize,
                 prop.maxThreadsPerBlock,
                 prop.maxThreadsPerMultiProcessor,
                 gpu_clock_khz / 1000,
                 mem_clock_khz / 1000,
                 prop.memoryBusWidth,
                 prop.l2CacheSize / 1024,
                 static_cast<unsigned long long>(prop.totalGlobalMem / (1024ULL * 1024ULL)),
                 static_cast<unsigned long long>(total_bytes ? free_bytes / (1024ULL * 1024ULL) : 0ULL),
                 static_cast<unsigned long long>(prop.sharedMemPerBlock / 1024ULL),
                 prop.ECCEnabled ? "yes" : "no",
                 prop.asyncEngineCount,
                 prop.concurrentKernels ? "yes" : "no",
                 prop.unifiedAddressing ? "yes" : "no",
                 prop.integrated ? "yes" : "no");
    }

    fmt::print(stderr,
               "[prach_detector_gpu] +======================================================================+\n");
  });
}

constexpr unsigned long long LOG_STATS_EVERY = 1000;

void record_gpu_detect_stats(unsigned long long&                   total_detects,
                             unsigned long long&                   total_graph_builds,
                             std::size_t                           cached_graphs,
                             unsigned long long&                   sum_detect_ns,
                             unsigned long long&                   max_detect_ns,
                             unsigned long long&                   min_detect_ns,
                             std::chrono::steady_clock::time_point start)
{
  auto ns = static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
  ++total_detects;
  sum_detect_ns += ns;
  if (ns > max_detect_ns) {
    max_detect_ns = ns;
  }
  if (min_detect_ns == 0 || ns < min_detect_ns) {
    min_detect_ns = ns;
  }
  if ((total_detects % LOG_STATS_EVERY) == 0) {
    fmt::print(stderr,
               "[prach_detector_gpu] stats: detects={} graph_builds={} cached_graphs={} "
               "mean={}us min={}us max={}us (last_window={})\n",
               total_detects,
               total_graph_builds,
               cached_graphs,
               (sum_detect_ns / LOG_STATS_EVERY) / 1000,
               min_detect_ns / 1000,
               max_detect_ns / 1000,
               LOG_STATS_EVERY);
    sum_detect_ns = 0;
    max_detect_ns = 0;
    min_detect_ns = 0;
  }
}

} // namespace

using namespace ocudu;

namespace {

constexpr unsigned MAX_LRA    = prach_constants::LONG_SEQUENCE_LENGTH;
constexpr unsigned MAX_DFT    = 4096;
constexpr unsigned MAX_SHIFTS = prach_constants::MAX_NUM_PREAMBLES;
constexpr unsigned MAX_PORTS  = 64;
constexpr unsigned MAX_SYMS   = 12;
constexpr unsigned MAX_PS     = MAX_PORTS * MAX_SYMS;
constexpr unsigned MAX_SEQ    = 64;

void check_cuda(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
  }
}

void check_cufft(cufftResult res, const char* what)
{
  if (res != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("cuFFT error in ") + what + ": code " + std::to_string(static_cast<int>(res)));
  }
}

using plan_key       = std::tuple<bool, unsigned, unsigned>;
using cufft_plan_map = std::map<plan_key, cufftHandle>;

cufftHandle get_or_create_plan(cufft_plan_map& cache,
                               cudaStream_t    stream,
                               bool            long_preamble,
                               unsigned        total_batch,
                               unsigned        dft_size)
{
  plan_key key = std::make_tuple(long_preamble, total_batch, dft_size);
  auto     it  = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  cufftHandle plan = 0;
  int         n    = static_cast<int>(dft_size);
  check_cufft(cufftPlanMany(&plan,
                            1,
                            &n,
                            nullptr,
                            1,
                            static_cast<int>(dft_size),
                            nullptr,
                            1,
                            static_cast<int>(dft_size),
                            CUFFT_C2C,
                            static_cast<int>(total_batch)),
              "cufftPlanMany");
  check_cufft(cufftSetStream(plan, stream), "cufftSetStream");
  cache[key] = plan;
  return plan;
}

struct graph_key {
  bool     long_preamble;
  unsigned nof_sequences;
  unsigned ports_x_symbols;
  unsigned dft_size;
  unsigned L_ra;
  unsigned nof_shifts;
  unsigned win_width;
  unsigned win_margin;
  unsigned combine_iters;

  bool operator<(const graph_key& o) const
  {
    return std::tie(long_preamble, nof_sequences, ports_x_symbols, dft_size, L_ra, nof_shifts, win_width, win_margin, combine_iters) <
           std::tie(o.long_preamble, o.nof_sequences, o.ports_x_symbols, o.dft_size, o.L_ra, o.nof_shifts, o.win_width, o.win_margin, o.combine_iters);
  }
};

using graph_cache_t = std::map<graph_key, cudaGraphExec_t>;

} // namespace

prach_detector_gpu_impl::prach_detector_gpu_impl(std::unique_ptr<prach_generator> generator_,
                                                 std::unique_ptr<prach_detector>  fallback_,
                                                 unsigned                         idft_long_size_,
                                                 unsigned                         idft_short_size_) :
  generator(std::move(generator_)),
  fallback(std::move(fallback_)),
  idft_long_size(idft_long_size_),
  idft_short_size(idft_short_size_)
{
  ocudu_assert(generator, "prach_detector_gpu_impl: generator must not be null.");
  ocudu_assert(fallback, "prach_detector_gpu_impl: fallback must not be null.");

  ensure_resources();

  fmt::print(stderr,
             "[prach_detector_gpu] constructed: idft_long={} idft_short={} max_batch={} "
             "device_buffers_mib={}\n",
             idft_long_size,
             idft_short_size,
             MAX_BATCH,
             (MAX_SEQ * MAX_PORTS * MAX_DFT * sizeof(float) * 2 +
              MAX_SEQ * MAX_PORTS * MAX_DFT * sizeof(float) +
              MAX_SEQ * MAX_DFT * sizeof(float) * 3) /
                 (1024ULL * 1024ULL));
}

prach_detector_gpu_impl::~prach_detector_gpu_impl()
{
  release_resources();
}

void prach_detector_gpu_impl::release_resources()
{
  if (resources_init) {
    auto* pcache = static_cast<cufft_plan_map*>(plan_cache);
    if (pcache) {
      for (auto& kv : *pcache) {
        cufftDestroy(kv.second);
      }
      delete pcache;
      plan_cache = nullptr;
    }
    cudaFree(d_root);
    cudaFree(d_combined);
    cudaFree(d_preamble);
    cudaFree(d_idft);
    cudaFree(d_mod_sq);
    cudaFree(d_mod_sq_combined);
    cudaFree(d_num);
    cudaFree(d_den);
    cudaFree(d_argmax_idx);
    cudaFree(d_argmax_val);
    cudaFree(d_num_at_argmax);
    cudaFree(d_window_starts);
    cudaFreeHost(h_root);
    cudaFreeHost(h_preamble);
    cudaFreeHost(h_argmax_idx);
    cudaFreeHost(h_argmax_val);
    cudaFreeHost(h_num_at_argmax);
    cudaFreeHost(h_window_starts);
    resources_init = false;
  }
  if (graph_cache) {
    auto* gcache = static_cast<graph_cache_t*>(graph_cache);
    for (auto& kv : *gcache) {
      cudaGraphExecDestroy(kv.second);
    }
    delete gcache;
    graph_cache = nullptr;
  }
  if (stream != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    stream = nullptr;
  }
}

void prach_detector_gpu_impl::ensure_resources()
{
  if (resources_init) {
    return;
  }

  cudaStream_t s = nullptr;
  check_cuda(cudaStreamCreate(&s), "cudaStreamCreate");
  stream = static_cast<void*>(s);

  plan_cache  = static_cast<void*>(new cufft_plan_map());
  graph_cache = static_cast<void*>(new graph_cache_t());

  check_cuda(cudaMalloc(&d_root, MAX_SEQ * MAX_LRA * sizeof(float) * 2), "cudaMalloc d_root");
  check_cuda(cudaMalloc(&d_combined, MAX_PS * MAX_LRA * sizeof(float) * 2), "cudaMalloc d_combined");
  check_cuda(cudaMalloc(&d_preamble, MAX_PS * MAX_LRA * sizeof(uint32_t)), "cudaMalloc d_preamble");

  constexpr size_t IDFT_BYTES = static_cast<size_t>(MAX_SEQ) * MAX_PORTS * MAX_DFT * sizeof(float) * 2;
  check_cuda(cudaMalloc(&d_idft, IDFT_BYTES), "cudaMalloc d_idft");
  constexpr size_t MODSQ_BYTES = static_cast<size_t>(MAX_SEQ) * MAX_PORTS * MAX_DFT * sizeof(float);
  check_cuda(cudaMalloc(&d_mod_sq, MODSQ_BYTES), "cudaMalloc d_mod_sq");
  constexpr size_t MODSQ_C_BYTES = static_cast<size_t>(MAX_SEQ) * MAX_DFT * sizeof(float);
  check_cuda(cudaMalloc(&d_mod_sq_combined, MODSQ_C_BYTES), "cudaMalloc d_mod_sq_combined");
  constexpr size_t NUMDEN_BYTES = static_cast<size_t>(MAX_SEQ) * MAX_SHIFTS * MAX_DFT * sizeof(float);
  check_cuda(cudaMalloc(&d_num, NUMDEN_BYTES), "cudaMalloc d_num");
  check_cuda(cudaMalloc(&d_den, NUMDEN_BYTES), "cudaMalloc d_den");
  check_cuda(cudaMalloc(&d_argmax_idx, MAX_SEQ * MAX_SHIFTS * sizeof(uint32_t)), "cudaMalloc d_argmax_idx");
  check_cuda(cudaMalloc(&d_argmax_val, MAX_SEQ * MAX_SHIFTS * sizeof(float)), "cudaMalloc d_argmax_val");
  check_cuda(cudaMalloc(&d_num_at_argmax, MAX_SEQ * MAX_SHIFTS * sizeof(float)), "cudaMalloc d_num_at_argmax");
  check_cuda(cudaMalloc(&d_window_starts, MAX_SHIFTS * sizeof(uint32_t)), "cudaMalloc d_window_starts");

  check_cuda(cudaMallocHost(&h_root, MAX_SEQ * MAX_LRA * sizeof(float) * 2), "cudaMallocHost h_root");
  check_cuda(cudaMallocHost(&h_preamble, MAX_PS * MAX_LRA * sizeof(uint32_t)), "cudaMallocHost h_preamble");
  check_cuda(cudaMallocHost(&h_argmax_idx, MAX_SEQ * MAX_SHIFTS * sizeof(uint32_t)), "cudaMallocHost h_argmax_idx");
  check_cuda(cudaMallocHost(&h_argmax_val, MAX_SEQ * MAX_SHIFTS * sizeof(float)), "cudaMallocHost h_argmax_val");
  check_cuda(cudaMallocHost(&h_num_at_argmax, MAX_SEQ * MAX_SHIFTS * sizeof(float)), "cudaMallocHost h_num_at_argmax");
  check_cuda(cudaMallocHost(&h_window_starts, MAX_SHIFTS * sizeof(uint32_t)), "cudaMallocHost h_window_starts");

  resources_init = true;
}

namespace {

void issue_chain(cudaStream_t s,
                 cufftHandle  plan,
                 void*        d_combined,
                 void*        d_preamble,
                 void*        d_root_all,
                 void*        d_idft,
                 void*        d_mod_sq,
                 void*        d_mod_sq_combined,
                 void*        d_num,
                 void*        d_den,
                 void*        d_argmax_idx,
                 void*        d_argmax_val,
                 void*        d_num_at_argmax,
                 void*        d_window_starts,
                 const void*  h_preamble,
                 const void*  h_root_all,
                 const void*  h_window_starts,
                 void*        h_argmax_idx,
                 void*        h_argmax_val,
                 void*        h_num_at_argmax,
                 unsigned     L_ra,
                 unsigned     dft_size,
                 unsigned     ports_x_symbols,
                 unsigned     nof_sequences,
                 unsigned     nof_shifts,
                 unsigned     win_width,
                 unsigned     win_margin,
                 float        scale,
                 float        window_scale,
                 unsigned     combine_iters)
{
  cudaMemcpyAsync(d_root_all,
                  h_root_all,
                  static_cast<size_t>(nof_sequences) * L_ra * sizeof(float) * 2,
                  cudaMemcpyHostToDevice,
                  s);

  cudaMemcpyAsync(d_window_starts, h_window_starts, nof_shifts * sizeof(uint32_t), cudaMemcpyHostToDevice, s);

  for (unsigned it = 0; it < combine_iters; ++it) {
    const uint8_t* h_src =
        static_cast<const uint8_t*>(h_preamble) + static_cast<size_t>(it) * ports_x_symbols * L_ra * sizeof(uint32_t);
    cudaMemcpyAsync(d_preamble,
                    h_src,
                    static_cast<size_t>(ports_x_symbols) * L_ra * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    s);
    ocudu::hal::cuda::launch_cbf16_to_cf_init_or_add(
        d_combined, d_preamble, L_ra, ports_x_symbols, (it == 0), s);
  }

  unsigned total_batch = nof_sequences * ports_x_symbols;
  if (!ocudu::hal::cuda::launch_fused_prach_idft(d_mod_sq,
                                                 d_combined,
                                                 d_root_all,
                                                 L_ra,
                                                 ports_x_symbols,
                                                 total_batch,
                                                 dft_size,
                                                 scale,
                                                 s)) {
    ocudu::hal::cuda::launch_prod_conj_bin_reorder(
        d_idft, d_combined, d_root_all, L_ra, dft_size, ports_x_symbols, nof_sequences, s);
    cufftExecC2C(plan, static_cast<cufftComplex*>(d_idft), static_cast<cufftComplex*>(d_idft), CUFFT_INVERSE);
    ocudu::hal::cuda::launch_modulus_square_normalize(d_mod_sq, d_idft, dft_size, total_batch, scale, s);
  }

  ocudu::hal::cuda::launch_combine_mod_sq(
      d_mod_sq_combined, d_mod_sq, ports_x_symbols, nof_sequences, dft_size, s);

  ocudu::hal::cuda::launch_per_shift_accumulate(d_num,
                                                d_den,
                                                d_mod_sq_combined,
                                                static_cast<const unsigned*>(d_window_starts),
                                                nof_shifts,
                                                nof_sequences,
                                                win_margin,
                                                win_width,
                                                dft_size,
                                                window_scale,
                                                s);

  ocudu::hal::cuda::launch_finalize_per_shift(
      d_argmax_idx, d_argmax_val, d_num_at_argmax, d_num, d_den, nof_shifts, nof_sequences, win_width, s);

  size_t result_count = static_cast<size_t>(nof_sequences) * nof_shifts;
  cudaMemcpyAsync(h_argmax_idx, d_argmax_idx, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_argmax_val, d_argmax_val, result_count * sizeof(float), cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(h_num_at_argmax, d_num_at_argmax, result_count * sizeof(float), cudaMemcpyDeviceToHost, s);
}

} // namespace

prach_detection_result prach_detector_gpu_impl::detect(const prach_buffer& input, const configuration& config)
{
  auto detect_t0 = std::chrono::steady_clock::now();

  ocudu_assert(config.start_preamble_index + config.nof_preamble_indices <= prach_constants::MAX_NUM_PREAMBLES,
               "PRACH preamble range out of bounds.");

  prach_preamble_information preamble_info = is_long_preamble(config.format)
                                                 ? get_prach_preamble_long_info(config.format)
                                                 : get_prach_preamble_short_info(config.format, config.ra_scs, false);

  interval<unsigned> preamble_indices(config.start_preamble_index,
                                      config.start_preamble_index + config.nof_preamble_indices);

  unsigned N_cs = prach_cyclic_shifts_get(config.ra_scs, config.restricted_set, config.zero_correlation_zone);
  ocudu_assert(N_cs != PRACH_CYCLIC_SHIFTS_RESERVED, "Reserved cyclic shift.");

  bool     long_pre = is_long_preamble(config.format);
  unsigned L_ra     = long_pre ? prach_constants::LONG_SEQUENCE_LENGTH : prach_constants::SHORT_SEQUENCE_LENGTH;
  unsigned dft_size = long_pre ? idft_long_size : idft_short_size;

  unsigned nof_shifts    = 1;
  unsigned nof_sequences = 64;
  if (N_cs != 0) {
    nof_shifts    = std::min(prach_constants::MAX_NUM_PREAMBLES, L_ra / N_cs);
    nof_sequences = divide_ceil(64, nof_shifts);
  }

  double sampling_rate_Hz = static_cast<double>(dft_size) * ra_scs_to_Hz(preamble_info.scs);
  double cp_duration      = preamble_info.cp_length.to_seconds();
  auto   cp_prach =
      static_cast<unsigned>(std::floor(cp_duration * static_cast<double>(L_ra) * ra_scs_to_Hz(preamble_info.scs)));

  unsigned win_width = std::min(N_cs, cp_prach);
  if (N_cs == 0) {
    win_width = cp_prach;
  }
  if (win_width == L_ra) {
    win_width -= 20;
  }
  win_width = (win_width * dft_size) / L_ra;

  detail::threshold_params th_params;
  th_params.nof_rx_ports          = config.nof_rx_ports;
  th_params.scs                   = config.ra_scs;
  th_params.format                = config.format;
  th_params.zero_correlation_zone = config.zero_correlation_zone;
  auto [threshold, combine_symbols, win_margin] = detail::get_threshold_and_margin(th_params);

  unsigned max_delay_samples = (N_cs == 0) ? cp_prach : std::min(std::max(N_cs, 1U) - 1U, cp_prach);
  max_delay_samples          = (max_delay_samples * dft_size) / L_ra;

  unsigned nof_symbols   = preamble_info.nof_symbols;
  unsigned i_td_occasion = 0;
  unsigned i_fd_occasion = 0;

  unsigned ports_x_symbols = combine_symbols ? config.nof_rx_ports : (config.nof_rx_ports * nof_symbols);
  unsigned combine_iters   = combine_symbols ? nof_symbols : 1;

  float rssi = 0.0F;
  for (unsigned i_port = 0; i_port != config.nof_rx_ports; ++i_port) {
    for (unsigned i_symbol = 0; i_symbol != nof_symbols; ++i_symbol) {
      rssi += ocuduvec::average_power(input.get_symbol(i_port, i_td_occasion, i_fd_occasion, i_symbol));
    }
  }
  rssi /= static_cast<float>(config.nof_rx_ports * nof_symbols);

  prach_detection_result result;
  result.rssi_dB         = convert_power_to_dB(rssi);
  result.time_resolution = phy_time_unit::from_seconds(1.0 / sampling_rate_Hz);
  result.time_advance_max =
      phy_time_unit::from_seconds(static_cast<double>(max_delay_samples) * 0.8 / sampling_rate_Hz);
  result.preambles.clear();

  if (!std::isnormal(rssi)) {
    record_gpu_detect_stats(total_detects,
                            total_graph_builds,
                            graph_cache ? static_cast<graph_cache_t*>(graph_cache)->size() : 0,
                            sum_detect_ns,
                            max_detect_ns,
                            min_detect_ns,
                            detect_t0);
    return result;
  }

  ocudu_assert((win_margin > 0) && (threshold > 0.0),
               "Window margin and threshold are not selected for the number of ports (i.e., {}) and the preamble "
               "format (i.e., {}).",
               config.nof_rx_ports,
               to_string(config.format));

  ensure_resources();
  cudaStream_t s           = static_cast<cudaStream_t>(stream);
  auto&        cache       = *static_cast<cufft_plan_map*>(plan_cache);
  unsigned     total_batch = nof_sequences * ports_x_symbols;
  cufftHandle  plan        = get_or_create_plan(cache, s, long_pre, total_batch, dft_size);

  bool roots_dirty = !roots_cached || cached_root_format != config.format ||
                     cached_root_index != config.root_sequence_index ||
                     cached_root_restricted_set != config.restricted_set ||
                     cached_root_zcz != config.zero_correlation_zone ||
                     cached_root_nof_sequences != nof_sequences || cached_root_nof_shifts != nof_shifts ||
                     cached_root_L_ra != L_ra;
  if (roots_dirty) {
    auto* h_root_cf = static_cast<cf_t*>(h_root);
    for (unsigned i_sequence = 0; i_sequence != nof_sequences; ++i_sequence) {
      prach_generator::configuration gen_cfg;
      gen_cfg.format                = config.format;
      gen_cfg.root_sequence_index   = config.root_sequence_index;
      gen_cfg.preamble_index        = i_sequence * nof_shifts;
      gen_cfg.restricted_set        = config.restricted_set;
      gen_cfg.zero_correlation_zone = config.zero_correlation_zone;
      span<const cf_t> root         = generator->generate(gen_cfg);
      std::memcpy(h_root_cf + i_sequence * L_ra, root.data(), L_ra * sizeof(cf_t));
    }
    cached_root_format         = config.format;
    cached_root_index          = config.root_sequence_index;
    cached_root_restricted_set = config.restricted_set;
    cached_root_zcz            = config.zero_correlation_zone;
    cached_root_nof_sequences  = nof_sequences;
    cached_root_nof_shifts     = nof_shifts;
    cached_root_L_ra           = L_ra;
    roots_cached               = true;
  }

  auto* h_ws_u32 = static_cast<uint32_t*>(h_window_starts);
  for (unsigned w = 0; w != nof_shifts; ++w) {
    h_ws_u32[w] = (dft_size - (N_cs * w * dft_size) / L_ra) % dft_size;
  }

  auto* h_p = static_cast<uint8_t*>(h_preamble);
  for (unsigned it = 0; it < combine_iters; ++it) {
    uint8_t* h_p_it = h_p + static_cast<size_t>(it) * ports_x_symbols * L_ra * sizeof(cbf16_t);
    if (combine_symbols) {
      for (unsigned i_port = 0; i_port != config.nof_rx_ports; ++i_port) {
        span<const cbf16_t> p = input.get_symbol(i_port, i_td_occasion, i_fd_occasion, it);
        std::memcpy(h_p_it + i_port * L_ra * sizeof(cbf16_t), p.data(), L_ra * sizeof(cbf16_t));
      }
    } else {
      for (unsigned i_port = 0; i_port != config.nof_rx_ports; ++i_port) {
        for (unsigned i_symbol = 0; i_symbol != nof_symbols; ++i_symbol) {
          unsigned            ps = i_port * nof_symbols + i_symbol;
          span<const cbf16_t> p  = input.get_symbol(i_port, i_td_occasion, i_fd_occasion, i_symbol);
          std::memcpy(h_p_it + ps * L_ra * sizeof(cbf16_t), p.data(), L_ra * sizeof(cbf16_t));
        }
      }
    }
  }

  float scale        = 1.0F / static_cast<float>(dft_size * L_ra);
  float window_scale = static_cast<float>(dft_size) / static_cast<float>(L_ra);

  graph_key key{long_pre, nof_sequences, ports_x_symbols, dft_size, L_ra, nof_shifts, win_width, win_margin, combine_iters};
  auto*           gcache = static_cast<graph_cache_t*>(graph_cache);
  cudaGraphExec_t gexec  = nullptr;
  auto            it_g   = gcache->find(key);
  if (it_g == gcache->end()) {
    auto build_start = std::chrono::steady_clock::now();
    cudaGraph_t g    = nullptr;
    check_cuda(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture");
    issue_chain(s,
                plan,
                d_combined,
                d_preamble,
                d_root,
                d_idft,
                d_mod_sq,
                d_mod_sq_combined,
                d_num,
                d_den,
                d_argmax_idx,
                d_argmax_val,
                d_num_at_argmax,
                d_window_starts,
                h_preamble,
                h_root,
                h_window_starts,
                h_argmax_idx,
                h_argmax_val,
                h_num_at_argmax,
                L_ra,
                dft_size,
                ports_x_symbols,
                nof_sequences,
                nof_shifts,
                win_width,
                win_margin,
                scale,
                window_scale,
                combine_iters);
    check_cuda(cudaStreamEndCapture(s, &g), "cudaStreamEndCapture");
    check_cuda(cudaGraphInstantiate(&gexec, g, nullptr, nullptr, 0), "cudaGraphInstantiate");
    cudaGraphDestroy(g);
    (*gcache)[key] = gexec;
    auto build_us  = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - build_start)
                        .count();
    ++total_graph_builds;
    fmt::print(stderr,
               "[prach_detector_gpu] graph cache miss #{}: long={} seq={} ports*sym={} dft={} "
               "shifts={} win_width={} build={}us (cached_graphs={})\n",
               total_graph_builds,
               long_pre,
               nof_sequences,
               ports_x_symbols,
               dft_size,
               nof_shifts,
               win_width,
               build_us,
               gcache->size());
  } else {
    gexec = it_g->second;
  }

  check_cuda(cudaGraphLaunch(gexec, s), "cudaGraphLaunch");
  check_cuda(cudaStreamSynchronize(s), "cudaStreamSynchronize");

  auto*    aidx = static_cast<const uint32_t*>(h_argmax_idx);
  auto*    aval = static_cast<const float*>(h_argmax_val);
  auto*    anum = static_cast<const float*>(h_num_at_argmax);
  unsigned power_normalization = config.nof_rx_ports * L_ra * nof_symbols;
  if (combine_symbols) {
    power_normalization *= nof_symbols;
  }
  for (unsigned i_sequence = 0; i_sequence != nof_sequences; ++i_sequence) {
    interval<unsigned> sequence_preambles(i_sequence * nof_shifts, (i_sequence + 1) * nof_shifts);
    if (!preamble_indices.overlaps(sequence_preambles)) {
      continue;
    }
    for (unsigned i_window = 0; i_window != nof_shifts; ++i_window) {
      unsigned preamble_index = i_sequence * nof_shifts + i_window;
      if (!preamble_indices.contains(preamble_index)) {
        continue;
      }
      unsigned out_idx = i_sequence * nof_shifts + i_window;
      unsigned delay   = aidx[out_idx];
      float    peak    = aval[out_idx];
      if ((delay < win_width) && (peak > threshold) &&
          (delay < static_cast<float>(max_delay_samples) * 0.8F)) {
        prach_detection_result::preamble_indication& info = result.preambles.emplace_back();
        info.preamble_index                               = preamble_index;
        info.time_advance     = phy_time_unit::from_seconds(static_cast<double>(delay) / sampling_rate_Hz);
        info.detection_metric = peak / threshold;
        float preamble_power  = anum[out_idx] / static_cast<float>(power_normalization);
        info.preamble_power_dB = convert_power_to_dB(preamble_power);
      }
    }
  }

  record_gpu_detect_stats(total_detects,
                          total_graph_builds,
                          static_cast<graph_cache_t*>(graph_cache)->size(),
                          sum_detect_ns,
                          max_detect_ns,
                          min_detect_ns,
                          detect_t0);
  return result;
}

namespace ocudu {

std::unique_ptr<prach_detector> create_prach_detector_gpu(std::unique_ptr<prach_generator> generator,
                                                          std::unique_ptr<prach_detector>  fallback,
                                                          unsigned                         idft_long_size,
                                                          unsigned                         idft_short_size)
{
  if (!generator || !fallback) {
    return nullptr;
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return nullptr;
  }
  log_gpu_info_once();
  try {
    return std::make_unique<prach_detector_gpu_impl>(std::move(generator),
                                                     std::move(fallback),
                                                     idft_long_size,
                                                     idft_short_size);
  } catch (const std::exception&) {
    return nullptr;
  }
}

} // namespace ocudu

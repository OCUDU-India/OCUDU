// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include <cstdint>
#include <cuda_runtime.h>
#include <cufftdx.hpp>
#include <mutex>

using namespace cufftdx;

namespace {

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void k_fused_prach_idft(float*                            d_mod_sq,
                            const typename FFT::value_type*   d_combined,
                            const typename FFT::value_type*   d_roots,
                            unsigned                          L_ra,
                            unsigned                          ports_x_symbols,
                            float                             scale)
{
  using complex_type = typename FFT::value_type;
  extern __shared__ __align__(alignof(float4)) unsigned char shared_raw[];
  auto* shared_mem = reinterpret_cast<complex_type*>(shared_raw);

  complex_type thread_data[FFT::storage_size];

  unsigned batch_idx = blockIdx.x;
  unsigned ps_idx    = batch_idx % ports_x_symbols;
  unsigned seq_idx   = batch_idx / ports_x_symbols;

  const complex_type* combined_b = d_combined + ps_idx * L_ra;
  const complex_type* root_b     = d_roots + seq_idx * L_ra;

  constexpr unsigned dft_size = cufftdx::size_of<FFT>::value;
  constexpr unsigned stride   = FFT::stride;
  constexpr unsigned ept      = FFT::elements_per_thread;

  unsigned half_lo = L_ra / 2 + 1;
  unsigned half_hi = L_ra / 2;

  unsigned index = threadIdx.x;
  for (unsigned i = 0; i < ept; ++i) {
    complex_type r;
    r.x = 0.f;
    r.y = 0.f;
    unsigned bin = index;
    if (bin < half_lo) {
      unsigned     src = bin + half_hi;
      complex_type a   = combined_b[src];
      complex_type b   = root_b[src];
      r.x              = a.x * b.x + a.y * b.y;
      r.y              = a.y * b.x - a.x * b.y;
    } else if (bin >= dft_size - half_hi) {
      unsigned     src = bin - (dft_size - half_hi);
      complex_type a   = combined_b[src];
      complex_type b   = root_b[src];
      r.x              = a.x * b.x + a.y * b.y;
      r.y              = a.y * b.x - a.x * b.y;
    }
    thread_data[i] = r;
    index += stride;
  }

  FFT().execute(thread_data, shared_mem);

  float* mod_sq_b = d_mod_sq + batch_idx * dft_size;
  index           = threadIdx.x;
  for (unsigned i = 0; i < ept; ++i) {
    complex_type v   = thread_data[i];
    mod_sq_b[index]  = (v.x * v.x + v.y * v.y) * scale;
    index += stride;
  }
}

template <unsigned N>
struct fused_runner {
  using FFT = decltype(Block() + Size<N>() + Type<fft_type::c2c>() + Direction<fft_direction::inverse>() +
                       Precision<float>() + SM<860>() + ElementsPerThread<8>() + FFTsPerBlock<1>());

  using complex_type = typename FFT::value_type;

  static cudaError_t launch(float*       d_mod_sq,
                            const void*  d_combined,
                            const void*  d_roots,
                            unsigned     L_ra,
                            unsigned     ports_x_symbols,
                            unsigned     total_batch,
                            float        scale,
                            cudaStream_t stream)
  {
    static std::once_flag once;
    static cudaError_t    setup_err = cudaSuccess;
    std::call_once(once, []() {
      if (FFT::shared_memory_size > 48 * 1024) {
        setup_err = cudaFuncSetAttribute(k_fused_prach_idft<FFT>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         FFT::shared_memory_size);
      }
    });
    if (setup_err != cudaSuccess) {
      return setup_err;
    }
    dim3 grid(total_batch, 1, 1);
    k_fused_prach_idft<FFT><<<grid, FFT::block_dim, FFT::shared_memory_size, stream>>>(
        d_mod_sq,
        static_cast<const complex_type*>(d_combined),
        static_cast<const complex_type*>(d_roots),
        L_ra,
        ports_x_symbols,
        scale);
    return cudaGetLastError();
  }
};

} // namespace

namespace ocudu {
namespace hal {
namespace cuda {

bool launch_fused_prach_idft(void*       d_mod_sq,
                             const void* d_combined,
                             const void* d_roots,
                             unsigned    L_ra,
                             unsigned    ports_x_symbols,
                             unsigned    total_batch,
                             unsigned    dft_size,
                             float       scale,
                             void*       stream)
{
  cudaStream_t s        = static_cast<cudaStream_t>(stream);
  float*       mod_sq_p = static_cast<float*>(d_mod_sq);
  switch (dft_size) {
    case 128:
      fused_runner<128>::launch(mod_sq_p, d_combined, d_roots, L_ra, ports_x_symbols, total_batch, scale, s);
      return true;
    case 256:
      fused_runner<256>::launch(mod_sq_p, d_combined, d_roots, L_ra, ports_x_symbols, total_batch, scale, s);
      return true;
    case 512:
      fused_runner<512>::launch(mod_sq_p, d_combined, d_roots, L_ra, ports_x_symbols, total_batch, scale, s);
      return true;
    case 1024:
      fused_runner<1024>::launch(mod_sq_p, d_combined, d_roots, L_ra, ports_x_symbols, total_batch, scale, s);
      return true;
    case 2048:
      fused_runner<2048>::launch(mod_sq_p, d_combined, d_roots, L_ra, ports_x_symbols, total_batch, scale, s);
      return true;
    case 4096:
      fused_runner<4096>::launch(mod_sq_p, d_combined, d_roots, L_ra, ports_x_symbols, total_batch, scale, s);
      return true;
    default:
      return false;
  }
}

} // namespace cuda
} // namespace hal
} // namespace ocudu

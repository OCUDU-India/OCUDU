// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ocudu/hal/cuda/prach_detector_gpu_kernel.h"

#include <cstdint>
#include <cufft.h>
#include <cuda_runtime.h>

namespace {

struct cbf16_dev {
  uint16_t real;
  uint16_t imag;
};

__device__ inline float bf16_to_float(uint16_t bf16)
{
  return __uint_as_float(static_cast<uint32_t>(bf16) << 16);
}

__device__ inline float2 cbf16_to_cf(const cbf16_dev v)
{
  return make_float2(bf16_to_float(v.real), bf16_to_float(v.imag));
}

__device__ inline bool device_isnormal(float x)
{
  return isfinite(x) && (fabsf(x) >= 1.17549435e-38F);
}

__global__ void k_cbf16_to_cf(float2* d_combined, const cbf16_dev* d_preamble, unsigned L_ra, bool init)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L_ra) {
    return;
  }
  unsigned         ps         = blockIdx.y;
  const cbf16_dev* preamble_b = d_preamble + ps * L_ra;
  float2*          combined_b = d_combined + ps * L_ra;
  float2           v          = cbf16_to_cf(preamble_b[i]);
  if (init) {
    combined_b[i] = v;
  } else {
    float2 acc    = combined_b[i];
    combined_b[i] = make_float2(acc.x + v.x, acc.y + v.y);
  }
}

__global__ void k_prod_conj_bin_reorder(float2*       d_idft_in,
                                        const float2* d_combined,
                                        const float2* d_roots,
                                        unsigned      L_ra,
                                        unsigned      dft_size,
                                        unsigned      ports_x_symbols)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dft_size) {
    return;
  }
  unsigned ps      = blockIdx.y;
  unsigned seq     = blockIdx.z;
  unsigned out_idx = (seq * ports_x_symbols + ps) * dft_size + i;

  unsigned half_lo = L_ra / 2 + 1;
  unsigned half_hi = L_ra / 2;
  unsigned src;
  bool     valid = false;
  if (i < half_lo) {
    src   = i + half_hi;
    valid = true;
  } else if (i >= dft_size - half_hi) {
    src   = i - (dft_size - half_hi);
    valid = true;
  }
  float2 r = make_float2(0.f, 0.f);
  if (valid) {
    float2 a = d_combined[ps * L_ra + src];
    float2 b = d_roots[seq * L_ra + src];
    r        = make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
  }
  d_idft_in[out_idx] = r;
}

__global__ void k_modulus_square_normalize(float* d_mod_sq, const float2* d_in, unsigned dft_size, float scale)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dft_size) {
    return;
  }
  unsigned      b     = blockIdx.y;
  const float2* in_b  = d_in + b * dft_size;
  float*        out_b = d_mod_sq + b * dft_size;
  float2        v     = in_b[i];
  out_b[i]            = (v.x * v.x + v.y * v.y) * scale;
}

__global__ void k_combine_mod_sq(float*       d_combined,
                                 const float* d_mod_sq,
                                 unsigned     ports_x_symbols,
                                 unsigned     dft_size)
{
  unsigned bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= dft_size) {
    return;
  }
  unsigned     seq  = blockIdx.y;
  const float* base = d_mod_sq + seq * ports_x_symbols * dft_size + bin;
  float        sum  = 0.f;
#pragma unroll 4
  for (unsigned ps = 0; ps < ports_x_symbols; ++ps) {
    sum += base[ps * dft_size];
  }
  d_combined[seq * dft_size + bin] = sum;
}

__global__ void k_per_shift_accumulate_combined(float*          d_num,
                                                float*          d_den,
                                                const float*    d_mod_sq_combined,
                                                const unsigned* d_window_starts,
                                                unsigned        win_margin,
                                                unsigned        win_width,
                                                unsigned        dft_size,
                                                float           window_scale,
                                                unsigned        nof_shifts)
{
  extern __shared__ float s_partial[];

  unsigned shift = blockIdx.x;
  unsigned seq   = blockIdx.y;
  unsigned tid   = threadIdx.x;
  unsigned bdim  = blockDim.x;

  unsigned window_start = d_window_starts[shift];
  unsigned i_start_ref  = (window_start + dft_size - win_margin) % dft_size;
  unsigned ref_len      = 2 * win_margin + win_width;

  const float* mod_sq_b = d_mod_sq_combined + seq * dft_size;

  float my_sum = 0.f;
  for (unsigned k = tid; k < ref_len; k += bdim) {
    unsigned idx = (i_start_ref + k) % dft_size;
    my_sum += mod_sq_b[idx];
  }
  s_partial[tid] = my_sum;
  __syncthreads();
  for (unsigned s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_partial[tid] += s_partial[tid + s];
    }
    __syncthreads();
  }
  float reference = s_partial[0];

  unsigned base = (seq * nof_shifts + shift) * win_width;
  for (unsigned i = tid; i < win_width; i += bdim) {
    float scaled = mod_sq_b[(window_start + i) % dft_size] * window_scale;
    float diff   = reference - scaled;
    if (!device_isnormal(diff)) {
      diff = 1e-9F;
    }
    d_num[base + i] = scaled;
    d_den[base + i] = diff;
  }
}

__global__ void k_finalize_per_shift_xseq(unsigned*    d_argmax_idx,
                                          float*       d_argmax_val,
                                          float*       d_num_at_argmax,
                                          const float* d_num,
                                          const float* d_den,
                                          unsigned     win_width,
                                          unsigned     nof_shifts)
{
  extern __shared__ unsigned char s_raw[];
  unsigned*                       s_idx = reinterpret_cast<unsigned*>(s_raw);
  float*                          s_val = reinterpret_cast<float*>(s_idx + blockDim.x);

  unsigned shift = blockIdx.x;
  unsigned seq   = blockIdx.y;
  unsigned tid   = threadIdx.x;
  unsigned bdim  = blockDim.x;
  unsigned base  = (seq * nof_shifts + shift) * win_width;

  unsigned best_i   = 0;
  float    best_val = -INFINITY;
  for (unsigned i = tid; i < win_width; i += bdim) {
    float n = d_num[base + i];
    float d = fabsf(d_den[base + i]);
    float m = n / d;
    if (m > best_val) {
      best_val = m;
      best_i   = i;
    }
  }
  s_idx[tid] = best_i;
  s_val[tid] = best_val;
  __syncthreads();
  for (unsigned s = bdim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_val[tid + s] > s_val[tid]) {
        s_val[tid] = s_val[tid + s];
        s_idx[tid] = s_idx[tid + s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    unsigned out_idx           = seq * nof_shifts + shift;
    unsigned idx               = s_idx[0];
    d_argmax_idx[out_idx]      = idx;
    d_argmax_val[out_idx]      = s_val[0];
    d_num_at_argmax[out_idx]   = d_num[base + idx];
  }
}

__global__ void k_zero_floats(float* d_buf, unsigned n)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_buf[i] = 0.f;
  }
}

} // namespace

namespace ocudu {
namespace hal {
namespace cuda {

namespace {
constexpr unsigned BLOCK_DIM_1D    = 128;
constexpr unsigned PER_SHIFT_BLOCK = 128;
constexpr unsigned FINALIZE_BLOCK  = 128;

unsigned grid_for(unsigned n, unsigned block) { return (n + block - 1) / block; }
} // namespace

void launch_prod_conj_bin_reorder(void*       d_idft_in,
                                  const void* d_combined,
                                  const void* d_roots,
                                  unsigned    L_ra,
                                  unsigned    dft_size,
                                  unsigned    ports_x_symbols,
                                  unsigned    nof_sequences,
                                  void*       stream)
{
  dim3 grid(grid_for(dft_size, BLOCK_DIM_1D), ports_x_symbols, nof_sequences);
  k_prod_conj_bin_reorder<<<grid, BLOCK_DIM_1D, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float2*>(d_idft_in),
      static_cast<const float2*>(d_combined),
      static_cast<const float2*>(d_roots),
      L_ra,
      dft_size,
      ports_x_symbols);
}

void launch_cbf16_to_cf_init_or_add(void*       d_combined,
                                    const void* d_preamble_cbf16,
                                    unsigned    L_ra,
                                    unsigned    ports_x_symbols,
                                    bool        init,
                                    void*       stream)
{
  dim3 grid(grid_for(L_ra, BLOCK_DIM_1D), ports_x_symbols, 1);
  k_cbf16_to_cf<<<grid, BLOCK_DIM_1D, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float2*>(d_combined),
      static_cast<const cbf16_dev*>(d_preamble_cbf16),
      L_ra,
      init);
}

void launch_modulus_square_normalize(void*       d_mod_sq,
                                     const void* d_idft_out,
                                     unsigned    dft_size,
                                     unsigned    total_batch,
                                     float       scale,
                                     void*       stream)
{
  dim3 grid(grid_for(dft_size, BLOCK_DIM_1D), total_batch, 1);
  k_modulus_square_normalize<<<grid, BLOCK_DIM_1D, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(d_mod_sq),
      static_cast<const float2*>(d_idft_out),
      dft_size,
      scale);
}

void launch_combine_mod_sq(void*       d_mod_sq_combined,
                           const void* d_mod_sq,
                           unsigned    ports_x_symbols,
                           unsigned    nof_sequences,
                           unsigned    dft_size,
                           void*       stream)
{
  dim3 grid(grid_for(dft_size, BLOCK_DIM_1D), nof_sequences, 1);
  k_combine_mod_sq<<<grid, BLOCK_DIM_1D, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(d_mod_sq_combined),
      static_cast<const float*>(d_mod_sq),
      ports_x_symbols,
      dft_size);
}

void launch_per_shift_accumulate(void*           d_num,
                                 void*           d_den,
                                 const void*     d_mod_sq_combined,
                                 const unsigned* d_window_starts,
                                 unsigned        nof_shifts,
                                 unsigned        nof_sequences,
                                 unsigned        win_margin,
                                 unsigned        win_width,
                                 unsigned        dft_size,
                                 float           window_scale,
                                 void*           stream)
{
  size_t shmem = PER_SHIFT_BLOCK * sizeof(float);
  dim3   grid(nof_shifts, nof_sequences, 1);
  k_per_shift_accumulate_combined<<<grid, PER_SHIFT_BLOCK, shmem, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(d_num),
      static_cast<float*>(d_den),
      static_cast<const float*>(d_mod_sq_combined),
      d_window_starts,
      win_margin,
      win_width,
      dft_size,
      window_scale,
      nof_shifts);
}

void launch_finalize_per_shift(void*       d_argmax_idx,
                               void*       d_argmax_val,
                               void*       d_num_at_argmax,
                               const void* d_num,
                               const void* d_den,
                               unsigned    nof_shifts,
                               unsigned    nof_sequences,
                               unsigned    win_width,
                               void*       stream)
{
  size_t shmem = FINALIZE_BLOCK * (sizeof(unsigned) + sizeof(float));
  dim3   grid(nof_shifts, nof_sequences, 1);
  k_finalize_per_shift_xseq<<<grid, FINALIZE_BLOCK, shmem, static_cast<cudaStream_t>(stream)>>>(
      static_cast<unsigned*>(d_argmax_idx),
      static_cast<float*>(d_argmax_val),
      static_cast<float*>(d_num_at_argmax),
      static_cast<const float*>(d_num),
      static_cast<const float*>(d_den),
      win_width,
      nof_shifts);
}

void launch_zero_floats(void* d_buf, unsigned n, void* stream)
{
  k_zero_floats<<<grid_for(n, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(d_buf), n);
}

} // namespace cuda
} // namespace hal
} // namespace ocudu

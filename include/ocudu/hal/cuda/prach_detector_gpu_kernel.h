// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include <cstdint>

namespace ocudu {
namespace hal {
namespace cuda {

void launch_prod_conj_bin_reorder(void*       d_idft_in,
                                  const void* d_combined,
                                  const void* d_roots,
                                  unsigned    L_ra,
                                  unsigned    dft_size,
                                  unsigned    ports_x_symbols,
                                  unsigned    nof_sequences,
                                  void*       stream);

void launch_cbf16_to_cf_init_or_add(void*       d_combined,
                                    const void* d_preamble_cbf16,
                                    unsigned    L_ra,
                                    unsigned    ports_x_symbols,
                                    bool        init,
                                    void*       stream);

void launch_modulus_square_normalize(void*       d_mod_sq,
                                     const void* d_idft_out,
                                     unsigned    dft_size,
                                     unsigned    total_batch,
                                     float       scale,
                                     void*       stream);

void launch_combine_mod_sq(void*       d_mod_sq_combined,
                           const void* d_mod_sq,
                           unsigned    ports_x_symbols,
                           unsigned    nof_sequences,
                           unsigned    dft_size,
                           void*       stream);

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
                                 void*           stream);

void launch_finalize_per_shift(void*       d_argmax_idx,
                               void*       d_argmax_val,
                               void*       d_num_at_argmax,
                               const void* d_num,
                               const void* d_den,
                               unsigned    nof_shifts,
                               unsigned    nof_sequences,
                               unsigned    win_width,
                               void*       stream);

void launch_zero_floats(void* d_buf, unsigned n, void* stream);

bool launch_fused_prach_idft(void*       d_mod_sq,
                             const void* d_combined,
                             const void* d_roots,
                             unsigned    L_ra,
                             unsigned    ports_x_symbols,
                             unsigned    total_batch,
                             unsigned    dft_size,
                             float       scale,
                             void*       stream);

} // namespace cuda
} // namespace hal
} // namespace ocudu

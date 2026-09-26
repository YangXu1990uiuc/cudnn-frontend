# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Scalar reduction lifecycle shared by prepared quantized SDPA hosts."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def _reset_amax_kernel(amax: cute.Pointer):
    amax.store(cutlass.Float32(0.0))


_reset_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _unscale_amax_kernel(amax: cute.Pointer, scale: cute.Pointer):
    amax.store(amax.load() / scale.load())


_unscale_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

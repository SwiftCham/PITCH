//
//  File.metal
//  MTL_Quant
//
//  Created by Benjamin Stacey on 24/06/2026.
//

#include <metal_stdlib>
#include "MTLQuantTypes.h"

using namespace metal;

static inline uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Random sign for element i: +1 if if pcg-hash is low, else set -1.
static inline float rand_sign(uint seed, uint i)
{
    return (pcg_hash(seed + i) & 1u) ? 1.0f : -1.0f;
}

kernel void turbo_encode(
    const device float*    input          [[ buffer(0) ]],
    device atomic_uint*    packed         [[ buffer(1) ]],
    device atomic_uint*    residual_bits  [[ buffer(2) ]],
    device TurboMeta*      meta_out       [[ buffer(3) ]],
    constant TurboParams&  params         [[ buffer(4) ]],
    uint                   tid            [[ thread_position_in_grid ]],
    threadgroup float*     tg_scratch     [[ threadgroup(0) ]],
    threadgroup float*     tg_min         [[ threadgroup(1) ]],
    threadgroup float*     tg_max         [[ threadgroup(2) ]])
{
    const uint  dim  = params.dim;
    const uint  bits = params.bits;
    const float levels = float((1u << bits) - 1u);

    threadgroup float tg_stats[2];

    // 1. Rotation: y = (H / sqrt(d)) * D * x
    tg_scratch[tid] = rand_sign(params.seed, tid) * input[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint len = 1u; len < dim; len <<= 1u) {
        float lo = tg_scratch[tid & ~len];
        float hi = tg_scratch[tid |  len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_scratch[tid] = (tid & len) ? (lo - hi) : (lo + hi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tg_scratch[tid] *= rsqrt(float(dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Thread 0: parralel min/max reduction
    tg_min[tid] = tg_scratch[tid];
    tg_max[tid] = tg_scratch[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            tg_min[tid] = min(tg_min[tid], tg_min[tid + stride]);
            tg_max[tid] = max(tg_max[tid], tg_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float scale = (tg_max[0] - tg_min[0]) / levels;
        if (scale == 0.0f) scale = 1.0f;          // constant guard
        tg_stats[0] = scale;
        tg_stats[1] = tg_min[0];
        meta_out->scale  = scale;
        meta_out->offset = tg_min[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale  = tg_stats[0];
    const float offset = tg_stats[1];

    // 3. Scalar quantisation
    const float rotated   = tg_scratch[tid];
    const uint  quantised = uint(clamp(round((rotated - offset) / scale),
                                       0.0f, levels));

    // 4. Residual sign bit
    const float dequantised = float(quantised) * scale + offset;
    const float residual    = rotated - dequantised;
    if (residual >= 0.0f) {
        atomic_fetch_or_explicit(&residual_bits[tid >> 5u],
                                 1u << (tid & 31u),
                                 memory_order_relaxed);
    }

    // 5. Thread 0: parralel RMS(residual) tree reduction
    tg_min[tid] = residual * residual;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            tg_min[tid] += tg_min[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        meta_out->residualScale = sqrt(tg_min[0] / float(dim));
        threadgroup_barrier(mem_flags::mem_threadgroup); //changed
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 6. Bit packing
    const uint bit_start = tid * bits;
    const uint word      = bit_start >> 5u;
    const uint off       = bit_start & 31u;
    atomic_fetch_or_explicit(&packed[word], quantised << off,
                             memory_order_relaxed);
    if (off + bits > 32u) {
        atomic_fetch_or_explicit(&packed[word + 1u], quantised >> (32u - off),
                                 memory_order_relaxed);
    }
}

//
//  TurboDecoder.metal
//  PITCH
//
//  Created by Benjamin Stacey on 24/06/2026.
//

#include <metal_stdlib>
#include "PITCHTypes.h"

using namespace metal;

static inline uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static inline float rand_sign(uint seed, uint i)
{
    return (pcg_hash(seed + i) & 1u) ? 1.0f : -1.0f;
}

// Read the `bits` bits at offset `bit_start` from little-endian 32-bit words.
static inline uint unpack_value(const device uint* buf, uint bit_start, uint bits)
{
    const uint word = bit_start >> 5u;
    const uint off  = bit_start & 31u;
    uint value = buf[word] >> off;
    if (off + bits > 32u) {
        value |= buf[word + 1u] << (32u - off);
    }
    return value & ((1u << bits) - 1u);
}

kernel void turbo_decode(
    const device uint*      packed         [[ buffer(0) ]],
    const device uint*      residual_bits  [[ buffer(1) ]],
    const device TurboMeta* meta           [[ buffer(2) ]],
    device float*           output         [[ buffer(3) ]],
    constant TurboParams&   params         [[ buffer(4) ]],
    uint                    tid            [[ thread_position_in_grid ]],
    threadgroup float*      tg_scratch     [[ threadgroup(0) ]])
{
    const uint dim  = params.dim;
    const uint bits = params.bits;

    // 1. Unpack + dequantise + residual correction
    const uint  quantised   = unpack_value(packed, tid * bits, bits);
    const float dequantised = float(quantised) * meta->scale + meta->offset;

    const bool  sign_bit = (residual_bits[tid >> 5u] >> (tid & 31u)) & 1u;
    const float value    = dequantised
                         + (sign_bit ? meta->residualScale : -meta->residualScale);

    // 2. Inverse rotation: x = D * (H / sqrt(d)) * y
    tg_scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint len = 1u; len < dim; len <<= 1u) {
        float lo = tg_scratch[tid & ~len];
        float hi = tg_scratch[tid |  len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_scratch[tid] = (tid & len) ? (lo - hi) : (lo + hi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[tid] = rand_sign(params.seed, tid) * tg_scratch[tid] * rsqrt(float(dim));
}

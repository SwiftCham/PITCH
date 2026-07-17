//
//  File.metal
//  MTL_Quant
//
//  Created by Benjamin Stacey on 24/06/2026.
//

#include <metal_stdlib>
#include "MTLQuantTypes.h"
using namespace metal;

constant float TWO_PI = 6.28318530717958647692f;

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

kernel void polar_decode(
    const device uint*      packed      [[ buffer(0) ]],
    const device PolarMeta* meta        [[ buffer(1) ]],
    device float*           output      [[ buffer(2) ]],
    constant PolarParams&   params      [[ buffer(3) ]],
    uint                    tid         [[ thread_position_in_grid ]],
    threadgroup float*      tg_scratch  [[ threadgroup(0) ]])
{
    const uint  dim    = params.dim;
    const uint  bits   = params.bits;
    const uint  pairs  = dim >> 1u;
    const float levels = float((1u << bits) - 1u);

    // 1. Unpack + dequantise + Cartesian reconstruction
    // remaining threads only rejoin for the inverse WHT.
    if (tid < pairs) {
        const uint  rq = unpack_value(packed, (2u * tid)      * bits, bits);
        const uint  tq = unpack_value(packed, (2u * tid + 1u) * bits, bits);

        const float rscale = (meta->magnitude > 0.0f) ? meta->magnitude : 1.0f;
        const float r      = float(rq) / levels * rscale;
        const float theta  = float(tq) / levels * TWO_PI;

        tg_scratch[2u * tid]      = r * cos(theta);
        tg_scratch[2u * tid + 1u] = r * sin(theta);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Inverse rotation: x = D * (H / sqrt(d)) * y
    for (uint len = 1u; len < dim; len <<= 1u) {
        float lo = tg_scratch[tid & ~len];
        float hi = tg_scratch[tid |  len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_scratch[tid] = (tid & len) ? (lo - hi) : (lo + hi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[tid] = rand_sign(params.seed, tid) * tg_scratch[tid] * rsqrt(float(dim));
}


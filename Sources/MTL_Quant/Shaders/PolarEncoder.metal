//
//  File 2.metal
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

static inline void pack_value(device atomic_uint* buf, uint slot, uint bits, uint value)
{
    const uint bit_start = slot * bits;
    const uint word      = bit_start >> 5u;
    const uint off       = bit_start & 31u;
    const uint mask      = (1u << bits) - 1u; //changed
    const uint val       = value & mask;//changed
    atomic_fetch_or_explicit(&buf[word], val << off, memory_order_relaxed);//changed
    if (off + bits > 32u) {
        atomic_fetch_or_explicit(&buf[word + 1u], val >> (32u - off),
                                 memory_order_relaxed);
    }
}

kernel void polar_encode(
    const device float*    input       [[ buffer(0) ]],
    device atomic_uint*    packed      [[ buffer(1) ]],
    device PolarMeta*      meta_out    [[ buffer(2) ]],
    constant PolarParams&  params      [[ buffer(3) ]],
    uint                   tid         [[ thread_position_in_grid ]],
    threadgroup float*     tg_scratch  [[ threadgroup(0) ]])
{
    const uint  dim    = params.dim;
    const uint  bits   = params.bits;
    const uint  pairs  = dim >> 1u;
    const float levels = float((1u << bits) - 1u);

    threadgroup float tg_rscale;   // broadcast: radius quantisation scale

    // 1. Rotation: y = (H / sqrt(d)) * D * x (identical to turbo_encode)
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

    // 2. Pairwise polar transform
    float r = 0.0f, theta = 0.0f;
    if (tid < pairs) {
        const float a = tg_scratch[2u * tid];
        const float b = tg_scratch[2u * tid + 1u];
        r     = sqrt(a * a + b * b);
        theta = atan2(b, a);
        if (theta < 0.0f) theta += TWO_PI;   // wrap to [0, 2pi)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // all pairs done reading
    if (tid < pairs) {
        tg_scratch[tid] = r;                 // stash radii for the max scan
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    
    // 3. Thread 0: magnitude = max radius
    for (uint stride = pairs >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            tg_scratch[tid] = max(tg_scratch[tid], tg_scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float rmax = tg_scratch[0];
        meta_out->magnitude = rmax;
        tg_rscale = (rmax > 0.0f) ? rmax : 1.0f;   // zero-vector guard
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Quantise + pack (slot 2i = radius, slot 2i+1 = angle)
    if (tid < pairs) {
        const uint rq = uint(clamp(round(r / tg_rscale * levels), 0.0f, levels));
        const uint tq = uint(clamp(round(theta / TWO_PI * levels), 0.0f, levels));
        pack_value(packed, 2u * tid,      bits, rq);
        pack_value(packed, 2u * tid + 1u, bits, tq);
    }
}

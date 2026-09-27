//
//  PITCHKernels.metal
//  PITCH
//
//
//  Batch layout: one threadgroup per vector, `dim` threads per threadgroup, one
//  coordinate per thread. Dispatch `count` threadgroups to process a whole batch.
//
//
//  Created by Benjamin Stacey on 23/09/2026.

#include <metal_stdlib>
using namespace metal;

// Mirrored by `BatchParams`
struct BatchParams {
    uint dim;    // power of two, 2...1024
    uint bits;   // 2...8
    uint seed;   // rotation seed
    uint count;  // vectors in the batch
};
static_assert(sizeof(BatchParams) == 16, "BatchParams must be 16 bytes to match Swift");

constant float TWO_PI = 6.28318530717958647692f;

static inline uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Random diagonal sign for coordinate i. Shared by every vector with the same seed.
static inline float rand_sign(uint seed, uint i)
{
    return (pcg_hash(seed + i) & 1u) ? 1.0f : -1.0f;
}

// Words of packed codes per vector: ceil(dim * bits / 32).
static inline uint words_per_vector(uint dim, uint bits)
{
    return (dim * bits + 31u) >> 5u;
}

// In-place unnormalised Walsh-Hadamard transform of s[0..dim)
static inline void wht_inplace(threadgroup float* s, uint tid, uint dim)
{
    for (uint len = 1u; len < dim; len <<= 1u) {
        const float lo = s[tid & ~len];
        const float hi = s[tid |  len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s[tid] = (tid & len) ? (lo - hi) : (lo + hi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Tree reductions over dim values. every thread gets the result
static inline float tg_sum(threadgroup float* s, uint tid, uint dim, float v)
{
    s[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s[tid] += s[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float total = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);   // all reads of s[0] done before reuse
    return total;
}

static inline float tg_max(threadgroup float* s, uint tid, uint dim, float v)
{
    s[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s[tid] = max(s[tid], s[tid + stride]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float result = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// Packs codes[0..dim) (each < 2^bits) into `words` little-endian uint32 words
// Caller must place a threadgroup barrier after writing `codes`
static inline void pack_codes(threadgroup const uint* codes, device uint* out,
                              uint tid, uint dim, uint bits, uint words)
{
    if (tid >= words) { return; }
    const uint word_start = tid * 32u;
    const uint first = word_start / bits;
    const uint last  = min((word_start + 31u) / bits, dim - 1u);
    uint w = 0u;
    for (uint i = first; i <= last; ++i) {
        const int shift = int(i * bits) - int(word_start);
        const uint c = codes[i];
        w |= (shift >= 0) ? (c << uint(shift)) : (c >> uint(-shift));
    }
    out[tid] = w;
}

// Reads code i (bits wide) from one vector's packed words.
static inline uint unpack_code(const device uint* buf, uint i, uint bits)
{
    const uint start = i * bits;
    const uint w     = start >> 5u;
    const uint off   = start & 31u;
    uint v = buf[w] >> off;
    if (off + bits > 32u) { v |= buf[w + 1u] << (32u - off); }
    return v & ((1u << bits) - 1u);
}

// The codebook is the Lloyd-Max quantiser for one coordinate of a uniform unit vector in R^d.
kernel void turbo_encode(
    const device float*    input    [[ buffer(0) ]],
    device uint*           codes    [[ buffer(1) ]],
    device float*          norms    [[ buffer(2) ]],
    constant float*        codebook [[ buffer(3) ]],
    constant BatchParams&  p        [[ buffer(4) ]],
    uint                   tid      [[ thread_position_in_threadgroup ]],
    uint                   vec      [[ threadgroup_position_in_grid ]],
    threadgroup float*     scratch  [[ threadgroup(0) ]],
    threadgroup uint*      tg_codes [[ threadgroup(1) ]])
{
    if (vec >= p.count) { return; }          // uniform across the threadgroup
    const uint dim   = p.dim;
    const uint bits  = p.bits;
    const uint words = words_per_vector(dim, bits);

    const float x    = input[vec * dim + tid];
    const float norm = sqrt(tg_sum(scratch, tid, dim, x * x));
    const float inv  = (norm > 0.0f) ? (1.0f / norm) : 0.0f;

    scratch[tid] = rand_sign(p.seed, tid) * x * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    wht_inplace(scratch, tid, dim);
    const float u = scratch[tid] * rsqrt(float(dim));

    // Nearest centroid = number of decision boundaries strictly below u (binary lifting)
    // Boundaries are midpoints of the ascending codebook; ties go to the lower code.
    const uint levels = 1u << bits;
    uint code = 0u;
    for (uint step = levels >> 1u; step > 0u; step >>= 1u) {
        const uint cand = code + step;
        if (u > 0.5f * (codebook[cand - 1u] + codebook[cand])) { code = cand; }
    }

    tg_codes[tid] = code;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pack_codes(tg_codes, codes + vec * words, tid, dim, bits, words);
    if (tid == 0u) { norms[vec] = norm; }
}

kernel void turbo_decode(
    const device uint*     codes    [[ buffer(0) ]],
    const device float*    norms    [[ buffer(1) ]],
    constant float*        codebook [[ buffer(2) ]],
    device float*          output   [[ buffer(3) ]],
    constant BatchParams&  p        [[ buffer(4) ]],
    uint                   tid      [[ thread_position_in_threadgroup ]],
    uint                   vec      [[ threadgroup_position_in_grid ]],
    threadgroup float*     scratch  [[ threadgroup(0) ]])
{
    if (vec >= p.count) { return; }
    const uint dim   = p.dim;
    const uint bits  = p.bits;
    const uint words = words_per_vector(dim, bits);

    scratch[tid] = codebook[unpack_code(codes + vec * words, tid, bits)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    wht_inplace(scratch, tid, dim);

    output[vec * dim + tid] = rand_sign(p.seed, tid) * scratch[tid] * rsqrt(float(dim)) * norms[vec];
}

// Encode: y = H D x / sqrt(d); pair p = (y_2p, y_2p+1) -> (r, theta).
// Code 2p = radius on [0, rmax] with 2^b - 1 steps; code 2p+1 = angle on 2^b circular bins.
kernel void polar_encode(
    const device float*    input    [[ buffer(0) ]],
    device uint*           codes    [[ buffer(1) ]],
    device float*          scales   [[ buffer(2) ]],
    constant BatchParams&  p        [[ buffer(3) ]],
    uint                   tid      [[ thread_position_in_threadgroup ]],
    uint                   vec      [[ threadgroup_position_in_grid ]],
    threadgroup float*     scratch  [[ threadgroup(0) ]],
    threadgroup uint*      tg_codes [[ threadgroup(1) ]])
{
    if (vec >= p.count) { return; }
    const uint dim   = p.dim;
    const uint bits  = p.bits;
    const uint words = words_per_vector(dim, bits);

    scratch[tid] = rand_sign(p.seed, tid) * input[vec * dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    wht_inplace(scratch, tid, dim);

    const float rs   = rsqrt(float(dim));
    const uint  base = tid & ~1u;                 // both threads of a pair see the same (a, c)
    const float a    = scratch[base] * rs;
    const float c    = scratch[base + 1u] * rs;
    const float r    = sqrt(a * a + c * c);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rmax = tg_max(scratch, tid, dim, r);
    const float safe = (rmax > 0.0f) ? rmax : 1.0f;

    const uint radius_levels = (1u << bits) - 1u;
    const uint angle_bins    = 1u << bits;          // circular: 0 and 2*pi coincide
    uint code;
    if ((tid & 1u) == 0u) {
        code = uint(clamp(floor(r / safe * float(radius_levels) + 0.5f), 0.0f, float(radius_levels)));
    } else {
        float theta = precise::atan2(c, a);
        if (theta < 0.0f) { theta += TWO_PI; }
        code = uint(floor(theta / TWO_PI * float(angle_bins) + 0.5f)) & (angle_bins - 1u);
    }

    tg_codes[tid] = code;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pack_codes(tg_codes, codes + vec * words, tid, dim, bits, words);
    if (tid == 0u) { scales[vec] = rmax; }
}

kernel void polar_decode(
    const device uint*     codes    [[ buffer(0) ]],
    const device float*    scales   [[ buffer(1) ]],
    device float*          output   [[ buffer(2) ]],
    constant BatchParams&  p        [[ buffer(3) ]],
    uint                   tid      [[ thread_position_in_threadgroup ]],
    uint                   vec      [[ threadgroup_position_in_grid ]],
    threadgroup float*     scratch  [[ threadgroup(0) ]])
{
    if (vec >= p.count) { return; }
    const uint dim   = p.dim;
    const uint bits  = p.bits;
    const device uint* v = codes + vec * words_per_vector(dim, bits);

    const uint  base  = tid & ~1u;
    const float r     = float(unpack_code(v, base, bits)) / float((1u << bits) - 1u) * scales[vec];
    const float theta = float(unpack_code(v, base + 1u, bits)) / float(1u << bits) * TWO_PI;
    scratch[tid] = (tid & 1u) ? r * precise::sin(theta) : r * precise::cos(theta);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    wht_inplace(scratch, tid, dim);

    output[vec * dim + tid] = rand_sign(p.seed, tid) * scratch[tid] * rsqrt(float(dim));
}

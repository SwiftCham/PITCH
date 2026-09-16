#pragma once

// Passed as constant buffer to TurboQuant kernel
typedef struct {
    uint32_t dim;
    uint32_t bits;
    uint32_t seed;
    uint32_t padding;   // 16byte allignment
} TurboParams;

// Written by turbo_encode, read by turbo_decode
typedef struct {
    float scale;
    float offset;
    float residualScale;
    float padding;
} TurboMeta;

// Passed as constant buffer to PolarQuant kernel
typedef struct {
    uint32_t dim;
    uint32_t bits;
    uint32_t seed;
    uint32_t padding;
} PolarParams;

// Written by polar_encode, read by polar_decode
typedef struct {
    float magnitude;
    float padding[3];
} PolarMeta;

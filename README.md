# PITCH. [![DOI](https://zenodo.org/badge/1264261940.svg)](https://doi.org/10.5281/zenodo.23065071)

![alt text](https://github.com/SwiftCham/PITCH/blob/main/PITCHLogo.png)

**Parallel, In-memory Transformer Compression Header**: KV-cache quantization for Apple Silicon, written in Swift and Metal.

PITCH compresses a transformer's key–value cache on the GPU. It implements three quantizers as hand-written Metal compute kernels behind a Swift API that works inside a model's own GPU buffers and command buffers:

- **TurboQuant** (Zandieh et al., ICLR 2026): random rotation, then a precomputed Lloyd–Max codebook per coordinate.
- **A single-level PolarQuant variant** (after Han et al., AISTATS 2026): rotation, then quantized radius and angle pairs.
- **Per-channel keys** (KIVI-style): keys quantized per channel over groups of tokens, with an fp16 range per channel and group.

Every kernel is tested for parity against an executable Python specification, so accuracy results measured on the specification hold for the library.

This repository accompanies the MSc thesis *PITCH: An Open-Source Metal Library for KV-Cache Quantization on Apple Silicon* (Atlantic Technological University, 2026).

## What the evaluation found

At 3.6× compression relative to fp16, on WikiText-2 perplexity:

| Model | TurboQuant, 4 bits | Per-channel keys, 4 bits |
|---|---|---|
| GPT-2 | +4.2% | +0.6% |
| Llama-3.2-1B | +6.1% | +2.3% |
| Qwen2.5-0.5B | 8.7-fold | +1.4% |
| Qwen2.5-1.5B | over 500-fold | +1.8% |

Lower reconstruction error did not mean better model quality. TurboQuant has the lowest reconstruction error, yet per-vector methods, including reproductions of llama.cpp's and MLX's KV formats, collapse on the Qwen models. The cause is the bias in Qwen's key projection: subtracting its rotated value exactly, at no storage cost, restores every per-vector method to within 1.3–4.3%.

**In practice:** for models whose key projection has a bias, such as Qwen, quantize keys per channel. For values, TurboQuant works well.

## Requirements

- A Mac with Apple Silicon
- Xcode 27 / Swift 6.3 or later

## Installation

Add PITCH with Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/SwiftCham/PITCH", from: "1.0.0")
],
targets: [
    .target(name: "YourTarget", dependencies: ["PITCH"])
]
```

## Usage

**Arrays in, arrays out.** Convenient for testing and offline compression:

```swift
import PITCH

let pitch = try PITCH()
let batch = try pitch.encode(vectors, dim: 64, bits: 4, method: .turboQuant)
let restored = try pitch.decode(batch)
```

**GPU-resident.** For use inside a model: work is encoded into your command buffer, on your buffers, with no CPU copies or waits. Buffers must belong to `pitch.device`; pass your model's device to `PITCH(device:)` to share it.

```swift
let cfg = try pitch.config(method: .turboQuant, dim: 64, bits: 4)

// append token t's keys (one vector per head)
try pitch.enqueueEncode(input: newKeys,
                        codes: kCodes, codesOffset: t * h * cfg.codeStride,
                        scales: kNorms, scalesOffset: t * h * 4,
                        count: h, config: cfg, commandBuffer: cb)

// decode the whole cache for attention
try pitch.enqueueDecode(codes: kCodes, scales: kNorms, output: keys,
                        count: (t + 1) * h, config: cfg, commandBuffer: cb)
```

**Per-channel keys.** Keys for one head, token-major, quantized in groups of 64 tokens:

```swift
let block = try pitch.encodePerChannel(keys, dim: 64, bits: 4)
let restoredKeys = try pitch.decodePerChannel(block)
```

The GPU-resident equivalents are `enqueuePerChannelEncode` and `enqueuePerChannelDecode`. Encode each group of tokens as it completes; the newest, partial group can be kept in full precision or re-encoded as it grows.

### Storage

| Mode | Stored bits per coordinate | At 4 bits, d = 64 |
|---|---|---|
| TurboQuant, PolarQuant variant | b + 32/d | 4.5 (3.56× smaller than fp16) |
| Per-channel keys | b + 32/G (group size G) | 4.5 with G = 64 |

## Limitations

- **Decoding the whole cache every step costs time linear in the context,** about 14 ms per token for Qwen2.5-0.5B at 32k tokens. At long contexts, decoding needs to move inside the attention kernel, which PITCH does not yet do.
- **Inputs and outputs are float32;** models that keep caches in fp16 or bf16 need a conversion.
- **The per-vector methods need a power-of-two head dimension up to 1024;** per-channel keys accept any dimension up to 1024.
- **PITCH has not been integrated into a running model;** end-to-end quality was measured in Python on the executable specification.
- **The bias-removal fix is not yet built into the kernels.**

## Repository layout

| Path | Contents |
|---|---|
| `Sources/PITCH/` | The library: Swift API, codecs and `Shaders/PITCHKernels.metal` |
| `Tests/PITCHTests/` | Test suite (114 cases), parity fixtures and benchmarks |
| `reference/` | Executable specification, evaluation scripts and results; see [`reference/README.md`](reference/README.md) |

## Testing

```
swift test
```

The benchmarks that produce the paper's timing data are skipped by default:

```
PITCH_RUN_BENCHMARKS=1 swift test -c release
```

## Reproducing the paper

[`reference/README.md`](reference/README.md) maps every table, figure and quoted number to the script and results file that produce it. The extracted KV vectors and attention blocks are attached to the [v1.0.0 release](https://github.com/SwiftCham/PITCH/releases/tag/v1.0.0); place them in `reference/` to rerun the evaluation.

## License

MIT. See [`LICENSE`](LICENSE).

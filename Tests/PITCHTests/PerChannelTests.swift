//
//  PerChannelTests.swift
//  PITCHTests
//
//

import Testing
import Foundation
import Metal
@testable import PITCH

private struct ChannelFixture: Decodable {
    struct Group: Decodable { let name: String; let dim: Int; let count: Int; let input: String }
    struct Case: Decodable {
        let group: String
        let bits: Int
        let group_size: Int
        let codes: String
        let ranges: String
        let reconstruction: String
        let mse: Double
    }
    let groups: [Group]
    let cases: [Case]

    static func load() throws -> ChannelFixture {
        let url = Bundle.module.url(forResource: "channel_parity", withExtension: "json", subdirectory: "Fixtures")
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("Fixtures/channel_parity.json")
        return try JSONDecoder().decode(ChannelFixture.self, from: Data(contentsOf: url))
    }
}

private func halfBits(base64: String) -> [UInt16] {
    guard let data = Data(base64Encoded: base64) else { return [] }
    var out = [UInt16](repeating: 0, count: data.count / 2)
    _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
    return out
}

private func meanSquaredError(_ a: [Float], _ b: [Float]) -> Double {
    zip(a, b).reduce(0.0) { $0 + Double($1.0 - $1.1) * Double($1.0 - $1.1) } / Double(a.count)
}

/// Gaussian keys with large near-constant offsets on four channels, like Qwen's key bias.
private func offsetKeys(count: Int, dim: Int, seed: UInt64) -> [Float] {
    var x = TestVectors.gaussian(count: count, dim: dim, seed: seed)
    let offsets: [Float] = [40, -25, 18, -12]
    for t in 0..<count { for c in 0..<min(4, dim) { x[t * dim + c] += offsets[c] } }
    return x
}

@Suite("Per-channel keys")
struct PerChannelTests {

    let q = PITCH.shared

    // MARK: Parity

    @Test func fixtureIsPresentAndComplete() throws {
        let fixture = try ChannelFixture.load()
        #expect(fixture.cases.count == 17)
        #expect(Set(fixture.cases.map(\.group_size)) == [16, 64])
    }

    @Test func metalMatchesReference() throws {
        let fixture = try ChannelFixture.load()
        let groups = Dictionary(uniqueKeysWithValues: fixture.groups.map { ($0.name, $0) })

        for c in fixture.cases {
            let g = try #require(groups[c.group])
            let label = "\(c.group) \(c.bits)-bit G=\(c.group_size)"
            let config = try ChannelQuantConfig(dim: g.dim, bits: c.bits, groupSize: c.group_size)
            let input = floats(base64: g.input)
            let refCodes = try #require(Data(base64Encoded: c.codes))
            let refRanges = halfBits(base64: c.ranges)
            let refRecon = floats(base64: c.reconstruction)

            // Encode: ranges are exact (min, max and fp16 rounding are deterministic);
            // codes may differ at a handful of float32 decision boundaries.
            let ours = try q.encodePerChannel(input, config: config)
            #expect(ours.ranges == refRanges, "\(label): fp16 ranges differ")
            let a = unpackCodes(ours.codes, count: g.count, dim: g.dim, bits: c.bits)
            let b = unpackCodes(refCodes, count: g.count, dim: g.dim, bits: c.bits)
            let mismatches = zip(a, b).filter { $0 != $1 }.count
            #expect(mismatches <= max(2, a.count / 500), "\(label): \(mismatches) of \(a.count) codes differ")

            // Decode the reference's codes.
            let refBlock = try ChannelCompressedBlock(config: config, codes: refCodes, ranges: refRanges)
            let decoded = try q.decodePerChannel(refBlock)
            let diff = ErrorMetrics.maxRelativeDifference(decoded, refRecon, dim: g.dim)
            #expect(diff < 1e-5, "\(label): decode differs from reference by \(diff)")

            // End-to-end error.
            let e = meanSquaredError(input, try q.decodePerChannel(ours))
            #expect(abs(e - c.mse) <= 0.02 * c.mse + 1e-9, "\(label): MSE \(e) vs reference \(c.mse)")
        }
    }

    // MARK: Behaviour

    @Test(arguments: [64, 96, 128])
    func errorWithinHalfAStep(dim: Int) throws {
        // |x - x̂| <= step / 2, plus the fp16 rounding of the stored range (relative 2^-11).
        let count = 200, bits = 4
        let x = offsetKeys(count: count, dim: dim, seed: UInt64(dim))
        let block = try q.encodePerChannel(x, dim: dim, bits: bits)
        let y = try q.decodePerChannel(block)
        for t in 0..<count {
            let g = t / block.config.groupSize
            for ch in 0..<dim {
                let r = block.range(group: g, channel: ch)
                let step = (r.max - r.min) / Float((1 << bits) - 1)
                let slack = 1e-3 * max(abs(r.min), abs(r.max)) + 1e-6
                #expect(abs(x[t * dim + ch] - y[t * dim + ch]) <= 0.5 * step + slack)
            }
        }
    }

    @Test func beatsTurboQuantOnOffsetKeysAtEqualBits() throws {
        // 4-bit per-channel (4.5 bits/coord at G = 64) against 4-bit TurboQuant (4.5 at d = 64).
        let x = offsetKeys(count: 256, dim: 64, seed: 3)
        let channel = try q.decodePerChannel(q.encodePerChannel(x, dim: 64, bits: 4))
        let turbo = try q.decode(q.encode(x, dim: 64, bits: 4, method: .turboQuant))
        #expect(meanSquaredError(x, channel) < 0.5 * meanSquaredError(x, turbo))
    }

    @Test func constantChannelIsExact() throws {
        let dim = 8
        var x = TestVectors.gaussian(count: 64, dim: dim, seed: 5)
        for t in 0..<64 { x[t * dim] = 1.5 }                        // exactly representable in fp16
        let y = try q.decodePerChannel(q.encodePerChannel(x, dim: dim, bits: 2))
        for t in 0..<64 { #expect(y[t * dim] == 1.5) }
    }

    @Test func partialFinalGroup() throws {
        let block = try q.encodePerChannel(offsetKeys(count: 150, dim: 64, seed: 9), dim: 64, bits: 4)
        #expect(block.count == 150)
        #expect(block.groupCount == 3)
        #expect(block.ranges.count == 3 * 2 * 64)
    }

    @Test func groupwiseGPUEncodeMatchesWholeBlock() throws {
        // A growing cache: encode each group separately at its offsets, then decode everything
        // in one call. Must equal encoding all tokens at once.
        let dim = 64, count = 150
        let config = try q.channelConfig(dim: dim, bits: 4)
        let x = offsetKeys(count: count, dim: dim, seed: 11)
        let device = q.device
        let input = makeBuffer(device, x)
        let codes = try #require(device.makeBuffer(length: count * config.codeStride, options: .storageModeShared))
        let groups = config.groupCount(tokens: count)
        let ranges = try #require(device.makeBuffer(length: groups * config.rangeStride, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: count * dim * 4, options: .storageModeShared))

        let cb = try q.makeCommandBuffer()
        for g in 0..<groups {
            let first = g * config.groupSize
            let n = min(config.groupSize, count - first)
            try q.enqueuePerChannelEncode(input: input, inputOffset: first * dim * 4,
                                          codes: codes, codesOffset: first * config.codeStride,
                                          ranges: ranges, rangesOffset: g * config.rangeStride,
                                          count: n, config: config, commandBuffer: cb)
        }
        try q.enqueuePerChannelDecode(codes: codes, ranges: ranges, output: output,
                                      count: count, config: config, commandBuffer: cb)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        let whole = try q.decodePerChannel(q.encodePerChannel(x, config: config))
        #expect(readFloats(output, count: count * dim) == whole)
    }

    @Test func appendRequiresGroupBoundary() throws {
        let x = offsetKeys(count: 64, dim: 16, seed: 2)
        var a = try q.encodePerChannel(x, dim: 16, bits: 4)
        try a.append(contentsOf: q.encodePerChannel(Array(x[0..<(10 * 16)]), dim: 16, bits: 4))
        #expect(a.count == 74 && a.groupCount == 2)
        #expect(throws: PITCHError.self) {
            try a.append(contentsOf: q.encodePerChannel(x, dim: 16, bits: 4))
        }
    }

    // MARK: Validation and storage

    @Test func rejectsInvalidArguments() throws {
        #expect(throws: PITCHError.self) { try ChannelQuantConfig(dim: 0, bits: 4) }
        #expect(throws: PITCHError.self) { try ChannelQuantConfig(dim: 2048, bits: 4) }
        #expect(throws: PITCHError.self) { try ChannelQuantConfig(dim: 64, bits: 1) }
        #expect(throws: PITCHError.self) { try ChannelQuantConfig(dim: 64, bits: 4, groupSize: 0) }
        #expect(throws: PITCHError.self) { try q.encodePerChannel([1, 2, 3], dim: 2, bits: 4) }
        #expect(throws: PITCHError.self) { try q.encodePerChannel([1, .nan], dim: 2, bits: 4) }
        #expect(throws: PITCHError.self) { try q.encodePerChannel([1, 70_000], dim: 2, bits: 4) }
        let config = try ChannelQuantConfig(dim: 4, bits: 4)
        #expect(throws: PITCHError.self) {                           // ranges too short
            try ChannelCompressedBlock(config: config, codes: Data(count: 4), ranges: [0, 0])
        }
        #expect(throws: PITCHError.self) {                           // min above max (2.0 > 1.0)
            try ChannelCompressedBlock(config: config, codes: Data(count: 4),
                                       ranges: [0x4000, 0, 0, 0, 0x3C00, 0, 0, 0])
        }
    }

    @Test func storageAccounting() throws {
        let c = try ChannelQuantConfig(dim: 64, bits: 4)
        #expect(c.bitsPerCoordinate == 4.5)                 // same as TurboQuant at d = 64
        #expect(try ChannelQuantConfig(dim: 128, bits: 4).bitsPerCoordinate == 4.5)
        #expect(try ChannelQuantConfig(dim: 128, bits: 4, groupSize: 128).bitsPerCoordinate == 4.25)
        #expect(abs(c.compressionRatioVsFloat16 - 16.0 / 4.5) < 1e-12)
        let block = try q.encodePerChannel(offsetKeys(count: 128, dim: 64, seed: 1), config: c)
        #expect(block.storedByteCount == 128 * 32 + 2 * 256 + ChannelCompressedBlock.headerByteCount)
    }

    @Test func halfConversion() {
        #expect(halfToFloat(0x3C00) == 1)
        #expect(halfToFloat(0xC000) == -2)
        #expect(halfToFloat(0x7BFF) == 65_504)
        #expect(halfToFloat(0x0001) == 0x1p-24)
        #expect(halfToFloat(0x7C00) == .infinity)
        #expect(halfToFloat(0x7E00).isNaN)
    }

    @Test func codableRoundTrip() throws {
        let block = try q.encodePerChannel(offsetKeys(count: 70, dim: 32, seed: 4), dim: 32, bits: 3, groupSize: 32)
        let back = try JSONDecoder().decode(ChannelCompressedBlock.self, from: JSONEncoder().encode(block))
        #expect(back == block)
    }
}

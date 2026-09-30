//
//  PerChannel.swift
//  PITCH
//
//  Per-channel (KIVI-style) key quantisation.
//
//  Created by Benjamin Stacey on 25/09/2026.
//

import Accelerate
import Foundation
import Metal


public struct ChannelQuantConfig: Sendable, Hashable, Codable {
    public static let supportedBits: ClosedRange<Int> = QuantConfig.supportedBits
    public static let maxDimension = QuantConfig.maxDimension
    public static let supportedGroupSizes: ClosedRange<Int> = 1...65_536
    public static let defaultGroupSize = 64
    // Channel minima and maxima are stored as fp16, so inputs must fit in its range
    public static let maxInputMagnitude: Float = 65_504

    //Channels per token (the head dimension). Any value 1...1024; no power-of-two requirement
    public let dim: Int
    // Bits per code, 2...8. One code per coordinate
    public let bits: Int
    // Tokens per group. Each group stores one fp16 minimum and maximum per channel
    public let groupSize: Int

    public init(dim: Int, bits: Int, groupSize: Int = ChannelQuantConfig.defaultGroupSize) throws {
        guard (1...Self.maxDimension).contains(dim) else {
            throw PITCHError.invalidInput("per-channel dimension \(dim) must be in 1...\(Self.maxDimension)")
        }
        guard Self.supportedBits.contains(bits) else {
            throw PITCHError.invalidBitWidth(bits)
        }
        guard Self.supportedGroupSizes.contains(groupSize) else {
            throw PITCHError.invalidInput("group size \(groupSize) must be in \(Self.supportedGroupSizes)")
        }
        self.dim = dim
        self.bits = bits
        self.groupSize = groupSize
    }

    private enum CodingKeys: String, CodingKey { case dim, bits, groupSize }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(dim: c.decode(Int.self, forKey: .dim),
                      bits: c.decode(Int.self, forKey: .bits),
                      groupSize: c.decode(Int.self, forKey: .groupSize))
    }

    // Bytes of packed codes per token (same packing as the per-vector methods).
    public var codeStride: Int { (dim * bits + 31) / 32 * 4 }

    // Bytes of fp16 ranges per group: `dim` minima then `dim` maxima
    public var rangeStride: Int { dim * 2 * MemoryLayout<UInt16>.size }

    // Groups needed for `tokens` tokens; the last may be partial
    public func groupCount(tokens: Int) -> Int { (tokens + groupSize - 1) / groupSize }

    //Storage cost per coordinate for full groups, including the ranges.
    public var bitsPerCoordinate: Double {
        Double(codeStride * 8) / Double(dim) + Double(rangeStride * 8) / Double(dim * groupSize)
    }

    public var compressionRatio: Double { 32.0 / bitsPerCoordinate }
    public var compressionRatioVsFloat16: Double { 16.0 / bitsPerCoordinate }
}

public struct ChannelCompressedBlock: Sendable, Equatable {
    public let config: ChannelQuantConfig
    public private(set) var codes: Data
    public private(set) var ranges: [UInt16]

    /// Number of tokens.
    public var count: Int { codes.count / config.codeStride }
    public var groupCount: Int { config.groupCount(tokens: count) }

    public init(config: ChannelQuantConfig, codes: Data, ranges: [UInt16]) throws {
        self.config = config
        self.codes = codes
        self.ranges = ranges
        try validate()
    }

    init(unchecked config: ChannelQuantConfig, codes: Data, ranges: [UInt16]) {
        self.config = config
        self.codes = codes
        self.ranges = ranges
    }

    public func validate() throws {
        guard !codes.isEmpty, codes.count % config.codeStride == 0 else {
            throw PITCHError.invalidInput(
                "codes is \(codes.count) bytes; expected a positive multiple of \(config.codeStride)")
        }
        let expected = groupCount * 2 * config.dim
        guard ranges.count == expected else {
            throw PITCHError.invalidInput(
                "ranges has \(ranges.count) values; expected \(expected) for \(count) tokens")
        }
        let d = config.dim
        for g in 0..<groupCount {
            for ch in 0..<d {
                let lo = halfToFloat(ranges[g * 2 * d + ch]), hi = halfToFloat(ranges[g * 2 * d + d + ch])
                guard lo.isFinite, hi.isFinite, lo <= hi else {
                    throw PITCHError.invalidInput("group \(g) channel \(ch): range [\(lo), \(hi)] is invalid")
                }
            }
        }
    }

    public func range(group: Int, channel: Int) -> (min: Float, max: Float) {
        let d = config.dim
        return (halfToFloat(ranges[group * 2 * d + channel]), halfToFloat(ranges[group * 2 * d + d + channel]))
    }

    public mutating func append(contentsOf other: ChannelCompressedBlock) throws {
        guard other.config == config else {
            throw PITCHError.configurationMismatch("cannot append a block with a different config")
        }
        guard count % config.groupSize == 0 else {
            throw PITCHError.invalidInput(
                "cannot append after a partial group (\(count) tokens, group size \(config.groupSize))")
        }
        codes.append(other.codes)
        ranges.append(contentsOf: other.ranges)
    }

    public static let headerByteCount = 16

    public var payloadByteCount: Int { codes.count }
    public var sideInformationByteCount: Int { ranges.count * MemoryLayout<UInt16>.size + Self.headerByteCount }
    public var storedByteCount: Int { payloadByteCount + sideInformationByteCount }
    public var originalByteCount: Int { count * config.dim * MemoryLayout<Float>.size }
    public var compressionRatio: Double { Double(originalByteCount) / Double(storedByteCount) }
    public var compressionRatioVsFloat16: Double { compressionRatio / 2 }
    public var bitsPerCoordinate: Double { Double(storedByteCount * 8) / Double(count * config.dim) }
}

extension ChannelCompressedBlock: Codable {
    public static let formatVersion = 1

    private enum CodingKeys: String, CodingKey { case formatVersion, config, codes, ranges }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let version = try c.decode(Int.self, forKey: .formatVersion)
        guard version == Self.formatVersion else {
            throw DecodingError.dataCorruptedError(
                forKey: .formatVersion, in: c,
                debugDescription: "Unsupported PITCH per-channel format version \(version); expected \(Self.formatVersion)")
        }
        try self.init(config: c.decode(ChannelQuantConfig.self, forKey: .config),
                      codes: c.decode(Data.self, forKey: .codes),
                      ranges: c.decode([UInt16].self, forKey: .ranges))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Self.formatVersion, forKey: .formatVersion)
        try c.encode(config, forKey: .config)
        try c.encode(codes, forKey: .codes)
        try c.encode(ranges, forKey: .ranges)
    }
}

// unavailable on Intel Macs. TODO: Implement fallback
func halfToFloat(_ h: UInt16) -> Float {
    let sign: Float = (h & 0x8000) != 0 ? -1 : 1
    let exponent = Int((h >> 10) & 0x1F)
    let mantissa = Float(h & 0x3FF)
    switch exponent {
    case 0:  return sign * mantissa * 0x1p-24                            // zero or subnormal
    case 31: return mantissa == 0 ? sign * .infinity : .nan
    default: return sign * (1 + mantissa / 1024) * Float(sign: .plus, exponent: exponent - 15, significand: 1)
    }
}

extension PITCH {

    public func channelConfig(dim: Int, bits: Int,
                              groupSize: Int = ChannelQuantConfig.defaultGroupSize) throws -> ChannelQuantConfig {
        try ChannelQuantConfig(dim: dim, bits: bits, groupSize: groupSize)
    }

    // Compresses `vectors.count / dim` tokens stored contiguously, token-major (one head's keys)
    public func encodePerChannel(_ vectors: [Float], dim: Int, bits: Int,
                                 groupSize: Int = ChannelQuantConfig.defaultGroupSize) throws -> ChannelCompressedBlock {
        try encodePerChannel(vectors, config: ChannelQuantConfig(dim: dim, bits: bits, groupSize: groupSize))
    }

    public func encodePerChannel(_ vectors: [Float], config: ChannelQuantConfig) throws -> ChannelCompressedBlock {
        guard !vectors.isEmpty, vectors.count % config.dim == 0 else {
            throw PITCHError.invalidInput("\(vectors.count) floats is not a whole number of \(config.dim)-dim tokens")
        }
        try Self.validateChannelValues(vectors)
        let count = vectors.count / config.dim
        let groups = config.groupCount(tokens: count)
        let device = manager.device

        guard let input = vectors.withUnsafeBytes({
                  device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let codes = device.makeBuffer(length: count * config.codeStride, options: .storageModeShared),
              let ranges = device.makeBuffer(length: groups * config.rangeStride, options: .storageModeShared)
        else { throw PITCHError.encodingFailed("could not allocate Metal buffers") }

        let cb = try makeCommandBuffer()
        try ChannelCodec(manager: manager).encode(input: input, inputOffset: 0, codes: codes, codesOffset: 0,
                                                  ranges: ranges, rangesOffset: 0,
                                                  count: count, config: config, commandBuffer: cb)
        try Self.runChannel(cb, failure: PITCHError.encodingFailed)

        let codeData = Data(bytes: codes.contents(), count: count * config.codeStride)
        let rangeCount = groups * 2 * config.dim
        let rangeArray = Array(UnsafeBufferPointer(
            start: ranges.contents().bindMemory(to: UInt16.self, capacity: rangeCount), count: rangeCount))
        return ChannelCompressedBlock(unchecked: config, codes: codeData, ranges: rangeArray)
    }

    // Reconstructs every token in the block, contiguously.
    public func decodePerChannel(_ block: ChannelCompressedBlock) throws -> [Float] {
        let n = block.count * block.config.dim
        guard let output = manager.device.makeBuffer(length: n * MemoryLayout<Float>.size,
                                                     options: .storageModeShared) else {
            throw PITCHError.decodingFailed("could not allocate Metal buffers")
        }
        try decodePerChannel(block, into: output, offset: 0)
        return Array(UnsafeBufferPointer(start: output.contents().bindMemory(to: Float.self, capacity: n), count: n))
    }

    // Reconstructs the block into `buffer` starting at byte `offset`, then waits
    public func decodePerChannel(_ block: ChannelCompressedBlock, into buffer: MTLBuffer, offset: Int = 0) throws {
        try block.validate()
        let device = manager.device
        guard let codes = block.codes.withUnsafeBytes({
                  device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let ranges = block.ranges.withUnsafeBytes({
                  device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        else { throw PITCHError.decodingFailed("could not allocate Metal buffers") }

        let cb = try makeCommandBuffer()
        try ChannelCodec(manager: manager).decode(codes: codes, codesOffset: 0, ranges: ranges, rangesOffset: 0,
                                                  output: buffer, outputOffset: offset,
                                                  count: block.count, config: block.config, commandBuffer: cb)
        try Self.runChannel(cb, failure: PITCHError.decodingFailed)
    }
    
    public func enqueuePerChannelEncode(input: MTLBuffer, inputOffset: Int = 0,
                                        codes: MTLBuffer, codesOffset: Int = 0,
                                        ranges: MTLBuffer, rangesOffset: Int = 0,
                                        count: Int, config: ChannelQuantConfig,
                                        commandBuffer: MTLCommandBuffer) throws {
        try ChannelCodec(manager: manager).encode(input: input, inputOffset: inputOffset,
                                                  codes: codes, codesOffset: codesOffset,
                                                  ranges: ranges, rangesOffset: rangesOffset,
                                                  count: count, config: config, commandBuffer: commandBuffer)
    }
    public func enqueuePerChannelDecode(codes: MTLBuffer, codesOffset: Int = 0,
                                        ranges: MTLBuffer, rangesOffset: Int = 0,
                                        output: MTLBuffer, outputOffset: Int = 0,
                                        count: Int, config: ChannelQuantConfig,
                                        commandBuffer: MTLCommandBuffer) throws {
        try ChannelCodec(manager: manager).decode(codes: codes, codesOffset: codesOffset,
                                                  ranges: ranges, rangesOffset: rangesOffset,
                                                  output: output, outputOffset: outputOffset,
                                                  count: count, config: config, commandBuffer: commandBuffer)
    }

    // Rejects NaN, infinity, and magnitudes beyond fp16's range.
    static func validateChannelValues(_ x: [Float]) throws {
        guard vDSP.sum(vDSP.multiply(Float(0), x)).isFinite else {
            throw PITCHError.invalidInput("input contains NaN or infinity")
        }
        let largest = vDSP.maximumMagnitude(x)
        guard largest <= ChannelQuantConfig.maxInputMagnitude else {
            throw PITCHError.invalidInput(
                "input magnitude \(largest) exceeds \(ChannelQuantConfig.maxInputMagnitude); per-channel ranges are stored in fp16")
        }
    }

    private static func runChannel(_ cb: MTLCommandBuffer, failure: (String) -> PITCHError) throws {
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw failure(error.localizedDescription) }
        guard cb.status == .completed else { throw failure("command buffer status \(cb.status.rawValue)") }
    }
}

// Mirrors `struct ChannelParams` in PITCHKernels.metal.
struct ChannelParams {
    var dim: UInt32
    var bits: UInt32
    var groupSize: UInt32
    var count: UInt32
}

struct ChannelCodec {
    let manager: DeviceManager

    init(manager: DeviceManager) {
        precondition(MemoryLayout<ChannelParams>.size == 16 && MemoryLayout<ChannelParams>.stride == 16,
                     "ChannelParams layout must match PITCHKernels.metal (16 bytes)")
        self.manager = manager
    }

    private func params(_ config: ChannelQuantConfig, count: Int) throws -> ChannelParams {
        guard count >= 1 else { throw PITCHError.invalidInput("count must be at least 1") }
        guard count * config.dim <= Int(UInt32.max) else {
            throw PITCHError.invalidInput("block too large: count * dim must fit in 32 bits")
        }
        return ChannelParams(dim: UInt32(config.dim), bits: UInt32(config.bits),
                             groupSize: UInt32(config.groupSize), count: UInt32(count))
    }

    private func check(_ buffer: MTLBuffer, offset: Int, needs bytes: Int, name: String) throws {
        guard buffer.device === manager.device else {
            throw PITCHError.invalidInput("\(name) buffer belongs to a different MTLDevice")
        }
        guard offset >= 0, offset % 4 == 0 else {
            throw PITCHError.invalidInput("\(name) offset \(offset) must be a non-negative multiple of 4")
        }
        guard buffer.length >= offset + bytes else {
            throw PITCHError.invalidInput("\(name) buffer has \(buffer.length - offset) bytes after offset; needs \(bytes)")
        }
    }

    private func pipeline(_ name: String, dim: Int) throws -> MTLComputePipelineState {
        let p = try manager.pipeline(named: name)
        guard dim <= p.maxTotalThreadsPerThreadgroup else {
            throw PITCHError.invalidInput("dimension \(dim) exceeds this GPU's \(p.maxTotalThreadsPerThreadgroup) threads per threadgroup")
        }
        return p
    }

    func encode(input: MTLBuffer, inputOffset: Int,
                codes: MTLBuffer, codesOffset: Int,
                ranges: MTLBuffer, rangesOffset: Int,
                count: Int, config: ChannelQuantConfig,
                commandBuffer: MTLCommandBuffer) throws {
        var p = try params(config, count: count)
        let groups = config.groupCount(tokens: count)
        try check(input, offset: inputOffset, needs: count * config.dim * 4, name: "input")
        try check(codes, offset: codesOffset, needs: count * config.codeStride, name: "codes")
        try check(ranges, offset: rangesOffset, needs: groups * config.rangeStride, name: "ranges")

        let pipeline = try self.pipeline("channel_encode", dim: config.dim)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PITCHError.encodingFailed("could not create compute encoder")
        }
        enc.label = "PITCH channel_encode"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: inputOffset, index: 0)
        enc.setBuffer(codes, offset: codesOffset, index: 1)
        enc.setBuffer(ranges, offset: rangesOffset, index: 2)
        enc.setBytes(&p, length: MemoryLayout<ChannelParams>.stride, index: 3)
        enc.setThreadgroupMemoryLength((config.dim * MemoryLayout<UInt32>.stride + 15) & ~15, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: config.dim, height: 1, depth: 1))
        enc.endEncoding()
    }

    func decode(codes: MTLBuffer, codesOffset: Int,
                ranges: MTLBuffer, rangesOffset: Int,
                output: MTLBuffer, outputOffset: Int,
                count: Int, config: ChannelQuantConfig,
                commandBuffer: MTLCommandBuffer) throws {
        var p = try params(config, count: count)
        let groups = config.groupCount(tokens: count)
        try check(codes, offset: codesOffset, needs: count * config.codeStride, name: "codes")
        try check(ranges, offset: rangesOffset, needs: groups * config.rangeStride, name: "ranges")
        try check(output, offset: outputOffset, needs: count * config.dim * 4, name: "output")

        let pipeline = try self.pipeline("channel_decode", dim: config.dim)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PITCHError.decodingFailed("could not create compute encoder")
        }
        enc.label = "PITCH channel_decode"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(codes, offset: codesOffset, index: 0)
        enc.setBuffer(ranges, offset: rangesOffset, index: 1)
        enc.setBuffer(output, offset: outputOffset, index: 2)
        enc.setBytes(&p, length: MemoryLayout<ChannelParams>.stride, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: config.dim, height: 1, depth: 1))
        enc.endEncoding()
    }
}

//
//  CompressedBatch.swift
//  PITCH
//
//  A batch of compressed vectors sharing one QuantConfig. Codes for vector i occupy
//  bytes [i * codeStride, (i + 1) * codeStride) of `codes`; its side information is
//  `scales[i]`
//
//  Created by Benjamin Stacey on 23/09/2026.

import Foundation

public struct CompressedBatch: Sendable, Equatable {
    public let config: QuantConfig
    public private(set) var codes: Data
    public private(set) var scales: [Float]

    // Number of vectors in the batch.
    public var count: Int { scales.count }

    // Throws if the sizes don't match `config` or a scale is negative or non-finite.
    public init(config: QuantConfig, codes: Data, scales: [Float]) throws {
        self.config = config
        self.codes = codes
        self.scales = scales
        try validate()
    }

    init(unchecked config: QuantConfig, codes: Data, scales: [Float]) {
        self.config = config
        self.codes = codes
        self.scales = scales
    }

    public func validate() throws {
        guard !scales.isEmpty else {
            throw PITCHError.invalidInput("a batch must contain at least one vector")
        }
        let expected = scales.count * config.codeStride
        guard codes.count == expected else {
            throw PITCHError.invalidInput(
                "codes is \(codes.count) bytes; expected \(expected) for \(scales.count) vectors")
        }
        guard scales.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw PITCHError.invalidInput("scales must be finite and non-negative")
        }
    }

    // Append another batch with the same configuration.
    public mutating func append(contentsOf other: CompressedBatch) throws {
        guard other.config == config else {
            throw PITCHError.configurationMismatch("cannot append a batch with a different config")
        }
        codes.append(other.codes)
        scales.append(contentsOf: other.scales)
    }

    // Puts the vectors in `range` as a new batch.
    public func vectors(_ range: Range<Int>) throws -> CompressedBatch {
        guard range.lowerBound >= 0, range.upperBound <= count, !range.isEmpty else {
            throw PITCHError.invalidInput("range \(range) is outside 0..<\(count)")
        }
        let stride = config.codeStride
        let start = codes.startIndex + range.lowerBound * stride
        let end = codes.startIndex + range.upperBound * stride
        return CompressedBatch(unchecked: config,
                               codes: Data(codes[start..<end]),
                               scales: Array(scales[range]))
    }

    // Serialised header: method, dim, bits, seed as four 32-bit values.
    public static let headerByteCount = 16

    // Packed codes.
    public var payloadByteCount: Int { codes.count }

    // One float per vector & per-batch header.
    public var sideInformationByteCount: Int {
        scales.count * MemoryLayout<Float>.size + Self.headerByteCount
    }

    public var storedByteCount: Int { payloadByteCount + sideInformationByteCount }

    public var originalByteCount: Int { count * config.dim * MemoryLayout<Float>.size }

    public var compressionRatio: Double { Double(originalByteCount) / Double(storedByteCount) }

    public var compressionRatioVsFloat16: Double { compressionRatio / 2 }

    public var bitsPerCoordinate: Double {
        Double(storedByteCount * 8) / Double(count * config.dim)
    }
}

extension CompressedBatch: Codable {
    // 3 = batched container, TurboQuant-MSE codebooks, single-level PolarQuant.
    public static let formatVersion = 3

    private enum CodingKeys: String, CodingKey { case formatVersion, config, codes, scales }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let version = try c.decode(Int.self, forKey: .formatVersion)
        guard version == Self.formatVersion else {
            throw DecodingError.dataCorruptedError(
                forKey: .formatVersion, in: c,
                debugDescription: "Unsupported PITCH format version \(version); expected \(Self.formatVersion)")
        }
        try self.init(config: c.decode(QuantConfig.self, forKey: .config),
                      codes: c.decode(Data.self, forKey: .codes),
                      scales: c.decode([Float].self, forKey: .scales))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Self.formatVersion, forKey: .formatVersion)
        try c.encode(config, forKey: .config)
        try c.encode(codes, forKey: .codes)
        try c.encode(scales, forKey: .scales)
    }
}

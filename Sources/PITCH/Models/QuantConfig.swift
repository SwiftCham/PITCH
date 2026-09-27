//
//  QuantConfig.swift
//  PITCH
//
//  Everything needed to interpret a batch of compressed vectors.
//
//  Created by Benjamin Stacey on 23/09/2026.

import Foundation

public struct QuantConfig: Sendable, Hashable, Codable {
    public static let supportedBits: ClosedRange<Int> = 2...8
    public static let maxDimension = 1024
    public static let maxInputMagnitude: Float = 1e17

    public let method: QuantMethod
    public let dim: Int
    // Bits per code, 2...8. TurboQuant: one code per coordinate.
    // PolarQuant: one radius code and one angle code per coordinate pair.
    public let bits: Int
    // Rotation seed
    public let seed: UInt32

    public init(method: QuantMethod, dim: Int, bits: Int, seed: UInt32 = PITCH.defaultSeed) throws {
        guard dim >= 2, dim <= Self.maxDimension, dim & (dim - 1) == 0 else {
            throw PITCHError.invalidDimension(dim)
        }
        guard Self.supportedBits.contains(bits) else {
            throw PITCHError.invalidBitWidth(bits)
        }
        self.method = method
        self.dim = dim
        self.bits = bits
        self.seed = seed
    }
    
    private enum CodingKeys: String, CodingKey { case method, dim, bits, seed }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(method: c.decode(QuantMethod.self, forKey: .method),
                      dim: c.decode(Int.self, forKey: .dim),
                      bits: c.decode(Int.self, forKey: .bits),
                      seed: c.decode(UInt32.self, forKey: .seed))
    }

    public var codeStride: Int { (dim * bits + 31) / 32 * 4 }

    // Bytes per vector including its one float of side information
    public var bytesPerVector: Int { codeStride + MemoryLayout<Float>.size }

    /// Storage cost per coordinate, including side information. The per-batch header is
    /// amortised away;
    public var bitsPerCoordinate: Double { Double(bytesPerVector * 8) / Double(dim) }

    // float32 size / compressed size, per vector.
    public var compressionRatio: Double { 32.0 / bitsPerCoordinate }

    /// float16 size / compressed size, per vector. 
    public var compressionRatioVsFloat16: Double { 16.0 / bitsPerCoordinate }
}

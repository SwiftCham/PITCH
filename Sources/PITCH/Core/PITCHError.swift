//
//  PITCHError.swift
//  PITCH
//
//  Created by Benjamin Stacey on 10/07/2026.
//

import Foundation

public enum PITCHError: Error, LocalizedError, Sendable, Equatable {
    case metalNotSupported
    case invalidInput(String)
    case invalidBitWidth(Int)
    case invalidDimension(Int)
    case configurationMismatch(String)
    case shaderCompilationFailed(String)
    case encodingFailed(String)
    case decodingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .metalNotSupported:
            return "Metal is not supported on this device."
        case .invalidInput(let msg):
            return "Invalid input: \(msg)"
        case .invalidBitWidth(let b):
            return "Invalid bit width \(b). Must be in \(QuantConfig.supportedBits)."
        case .invalidDimension(let d):
            return "Invalid dimension \(d). Must be a power of two in [2, \(QuantConfig.maxDimension)]."
        case .configurationMismatch(let msg):
            return "Configuration mismatch: \(msg)"
        case .shaderCompilationFailed(let msg):
            return "Failed to prepare Metal shader: \(msg)"
        case .encodingFailed(let msg):
            return "Encoding failed: \(msg)"
        case .decodingFailed(let msg):
            return "Decoding failed: \(msg)"
        }
    }
}

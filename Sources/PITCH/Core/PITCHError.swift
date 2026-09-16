//
//  PITCHError.swift
//  PITCH
//
//  Created by Benjamin Stacey on 10/07/2026.
//

import Foundation

public enum PITCHError: Error, LocalizedError, Sendable {
    case metalNotSupported
    case invalidInput(String)
    case invalidBitWidth(Int)
    case shaderCompilationFailed(String)
    case metadataMismatch
    case encodingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .metalNotSupported:
            return "Metal is not supported on this device."
        case .invalidInput(let msg):
            return "Invalid input: \(msg)"
        case .invalidBitWidth(let b):
            return "Invalid bit width \(b). Must be 3, 4, or 8."
        case .shaderCompilationFailed(let name):
            return "Failed to compile Metal shader: \(name)"
        case .metadataMismatch:
            return "Compressed metadata type does not match the requested decoder."
        case .encodingFailed(let msg):
            return "Encoding failed: \(msg)"
        }
    }
}

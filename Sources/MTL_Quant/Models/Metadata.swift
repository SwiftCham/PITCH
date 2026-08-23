//
//  Metadata.swift
//  MTL_Quant
//
//  Created by Benjamin Stacey on 10/07/2026.
//

import Foundation

public protocol Metadata: Sendable {
    var method: QuantMethod { get }
    var bits: Int { get }
    var dim: Int { get }
    var seed: UInt32 { get }
}

public struct TurboMetadata: Metadata {
    public let method: QuantMethod = .turboQuant
    public let bits: Int
    public let dim: Int
    public let seed: UInt32
    public let scale: Float
    public let offset: Float
    public let residualScale: Float
    public let residualData: Data
}

public struct PolarMetadata: Metadata {
    public let method: QuantMethod = .polarQuant
    public let bits: Int
    public let dim: Int
    public let seed: UInt32
    public let magnitude: Float
}

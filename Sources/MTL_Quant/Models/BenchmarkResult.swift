//
//  BenchmarkResult.swift
//  MTL_Quant
//
//  Created by Benjamin Stacey on 10/07/2026.
//

public struct BenchmarkResult: Sendable {
    public let algorithm: Method
    public let bits: Int
    public let dim: Int
    public let mse: Float
    public let innerProductError: Float
    public let encodeThroughputGBs: Float
    public let decodeThroughputGBs: Float
    public let compressionRatio: Float
}

//
//  TestSupport.swift
//  PITCHTests
//
//  Deterministic data, error metrics and small utilities shared by every suite.
//  Accuracy thresholds in the suites come from reference/pitch_reference.py run over
//  300 random trials; each sits above the worst trial with at least a 12% margin.
//

import Foundation
import Metal
@testable import PITCH

struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed &+ 0x9E37_79B9_7F4A_7C15
        if state == 0 { state = 1 }
        for _ in 0..<8 { _ = next() }
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

enum TestVectors {
    static func uniform(count: Int, dim: Int, seed: UInt64) -> [Float] {
        var g = SeededGenerator(seed: seed)
        return (0..<(count * dim)).map { _ in Float.random(in: -1...1, using: &g) }
    }
    static func gaussian(count: Int, dim: Int, seed: UInt64) -> [Float] {
        var g = SeededGenerator(seed: seed)
        let n = count * dim
        var out = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            let u1 = Float.random(in: Float.leastNormalMagnitude...1, using: &g)
            let u2 = Float.random(in: 0...1, using: &g)
            let r = (-2 * log(u1)).squareRoot()
            out[i] = r * cos(2 * .pi * u2)
            if i + 1 < n { out[i + 1] = r * sin(2 * .pi * u2) }
            i += 2
        }
        return out
    }
}

enum ErrorMetrics {
    static func meanRelativeError(_ x: [Float], _ xHat: [Float], dim: Int) -> Double {
        precondition(x.count == xHat.count && x.count % dim == 0)
        var total = 0.0, n = 0
        for v in 0..<(x.count / dim) {
            var num = 0.0, den = 0.0
            for i in (v * dim)..<((v + 1) * dim) {
                let d = Double(x[i]) - Double(xHat[i])
                num += d * d
                den += Double(x[i]) * Double(x[i])
            }
            if den > 0 { total += num / den; n += 1 }
        }
        return n > 0 ? total / Double(n) : 0
    }

    static func meanCosine(_ x: [Float], _ xHat: [Float], dim: Int) -> Double {
        var total = 0.0
        let count = x.count / dim
        for v in 0..<count {
            var dot = 0.0, a = 0.0, b = 0.0
            for i in (v * dim)..<((v + 1) * dim) {
                dot += Double(x[i]) * Double(xHat[i])
                a += Double(x[i]) * Double(x[i])
                b += Double(xHat[i]) * Double(xHat[i])
            }
            total += (a * b > 0) ? dot / (a * b).squareRoot() : 0
        }
        return total / Double(count)
    }
    static func maxRelativeDifference(_ a: [Float], _ b: [Float], dim: Int) -> Double {
        var worst = 0.0
        for v in 0..<(a.count / dim) {
            var num = 0.0, den = 0.0
            for i in (v * dim)..<((v + 1) * dim) {
                let d = Double(a[i]) - Double(b[i])
                num += d * d
                den += Double(b[i]) * Double(b[i])
            }
            worst = max(worst, den > 0 ? (num / den).squareRoot() : num.squareRoot())
        }
        return worst
    }
}

let testRotationSeed: UInt32 = 0x00C0_FFEE

func roundTripError(_ q: PITCH, method: QuantMethod, bits: Int, dim: Int, count: Int = 64,
                    gaussian: Bool = false, dataSeed: UInt64 = 1) throws -> Double {
    let x = gaussian ? TestVectors.gaussian(count: count, dim: dim, seed: dataSeed)
                     : TestVectors.uniform(count: count, dim: dim, seed: dataSeed)
    let batch = try q.encode(x, dim: dim, bits: bits, method: method, seed: testRotationSeed)
    return ErrorMetrics.meanRelativeError(x, try q.decode(batch), dim: dim)
}

func unpackCodes(_ data: Data, count: Int, dim: Int, bits: Int) -> [UInt32] {
    var words = [UInt32](repeating: 0, count: data.count / 4)
    _ = words.withUnsafeMutableBytes { data.copyBytes(to: $0) }     // alignment-safe copy
    let wordsPerVector = (dim * bits + 31) / 32
    let mask = UInt32((1 << bits) - 1)
    var out = [UInt32](); out.reserveCapacity(count * dim)
    for v in 0..<count {
        let base = v * wordsPerVector
        for i in 0..<dim {
            let start = i * bits, w = base + start / 32, off = UInt32(start % 32)
            var value = words[w] >> off
            if Int(off) + bits > 32 { value |= words[w + 1] << (32 - off) }
            out.append(value & mask)
        }
    }
    return out
}

/// Little-endian float32 array from base64.
func floats(base64: String) -> [Float] {
    guard let data = Data(base64Encoded: base64) else { return [] }
    var out = [Float](repeating: 0, count: data.count / 4)
    _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
    return out
}

/// Shared-storage buffer holding `values`.
func makeBuffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer {
    values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}

func readFloats(_ buffer: MTLBuffer, offset: Int = 0, count: Int) -> [Float] {
    let p = (buffer.contents() + offset).bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: p, count: count))
}

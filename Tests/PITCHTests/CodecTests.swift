//
//  CodecTests.swift
//  PITCHTests
//
//  Accuracy and behaviour of the two codecs through the array API.
//

import Testing
import Foundation
@testable import PITCH

@Suite("Codec accuracy")
struct CodecAccuracyTests {

    let q = PITCH.shared

    struct Case: Sendable, CustomTestStringConvertible {
        let method: QuantMethod
        let bits: Int
        let threshold: Double
        var testDescription: String { "\(method.rawValue) \(bits)-bit < \(threshold)" }
    }

    // Mean relative error over 64 U[-1,1] vectors. Reference means (d=64 / d=512):
    // TurboQuant 0.116/0.117, 0.034/0.035, 0.0092/0.0095, 4.1e-5/4.1e-5
    // PolarQuant 0.238/0.259, 0.058/0.062, 0.0143/0.0151, 5.5e-5/5.8e-5
    static let cases: [Case] = [
        Case(method: .turboQuant, bits: 2, threshold: 0.14),
        Case(method: .turboQuant, bits: 3, threshold: 0.045),
        Case(method: .turboQuant, bits: 4, threshold: 0.013),
        Case(method: .turboQuant, bits: 8, threshold: 1.5e-4),   // heavy per-vector tail at 8 bits
        Case(method: .polarQuant, bits: 2, threshold: 0.30),
        Case(method: .polarQuant, bits: 3, threshold: 0.075),
        Case(method: .polarQuant, bits: 4, threshold: 0.02),
        Case(method: .polarQuant, bits: 8, threshold: 7.5e-5),
    ]

    @Test(arguments: CodecAccuracyTests.cases, [64, 512])
    func roundTrip(_ c: Case, dim: Int) throws {
        let e = try roundTripError(q, method: c.method, bits: c.bits, dim: dim)
        #expect(e < c.threshold, "relative error \(e)")
    }

    /// TurboQuant's Theorem 1: 4^-b <= E||x - x̂||²/||x||² <= (sqrt(3) pi / 2) 4^-b.
    @Test(arguments: Array(QuantConfig.supportedBits))
    func turboQuantMeetsTheorem1(bits: Int) throws {
        let e = try roundTripError(q, method: .turboQuant, bits: bits, dim: 64, count: 16_384, gaussian: true)
        let lower = pow(4.0, -Double(bits))
        let upper = 3.0.squareRoot() * .pi / 2 * lower
        #expect(e >= lower && e <= upper,
                "error \(e) outside [\(lower), \(upper)] (\(Int(100 * (e - lower) / (upper - lower)))% of gap)")
    }

    @Test(arguments: QuantMethod.allCases)
    func errorFallsWithBits(method: QuantMethod) throws {
        let e = try [2, 3, 4, 8].map { try roundTripError(q, method: method, bits: $0, dim: 256) }
        #expect(e[0] > 2 * e[1] && e[1] > 2 * e[2], "2/3/4-bit errors \(e)")
        #expect(e[2] > 50 * e[3], "4-bit \(e[2]) vs 8-bit \(e[3])")
    }

    @Test func polarAngleCodebookUsesAllCodes() throws {
        let e = try roundTripError(q, method: .polarQuant, bits: 3, dim: 1024, count: 16, gaussian: true)
        #expect(e < 0.070, "relative error \(e)")
    }

    @Test(arguments: QuantMethod.allCases, [2, 1024])
    func extremeDimensionsRoundTrip(method: QuantMethod, dim: Int) throws {
        let x = TestVectors.gaussian(count: 8, dim: dim, seed: 3)
        let out = try q.decode(q.encode(x, dim: dim, bits: 8, method: method))
        #expect(out.count == x.count)
        #expect(out.allSatisfy { $0.isFinite })
        #expect(ErrorMetrics.meanRelativeError(x, out, dim: dim) < 0.01)
    }
}

@Suite("Codec behaviour")
struct CodecBehaviourTests {

    let q = PITCH.shared

    @Test(arguments: QuantMethod.allCases)
    func zeroVectorDecodesToZero(method: QuantMethod) throws {
        let out = try q.decode(q.encode([Float](repeating: 0, count: 64), bits: 4, method: method))
        #expect(out.allSatisfy { $0 == 0 })
    }

    @Test(arguments: QuantMethod.allCases)
    func scaleInvariance(method: QuantMethod) throws {
        // Both codecs normalise per vector, so scaling the input scales the output.
        let x = TestVectors.gaussian(count: 4, dim: 128, seed: 4)
        let a = try q.decode(q.encode(x, dim: 128, bits: 4, method: method))
        let b = try q.decode(q.encode(x.map { $0 * 1000 }, dim: 128, bits: 4, method: method))
        #expect(ErrorMetrics.maxRelativeDifference(b.map { $0 / 1000 }, a, dim: 128) < 1e-4)
    }

    @Test(arguments: QuantMethod.allCases)
    func sameSeedIsDeterministic(method: QuantMethod) throws {
        let x = TestVectors.uniform(count: 8, dim: 256, seed: 5)
        let a = try q.encode(x, dim: 256, bits: 4, method: method, seed: 42)
        let b = try q.encode(x, dim: 256, bits: 4, method: method, seed: 42)
        #expect(a == b)
    }

    @Test(arguments: QuantMethod.allCases)
    func seedChangesTheRotation(method: QuantMethod) throws {
        let x = TestVectors.uniform(count: 8, dim: 256, seed: 6)
        let a = try q.encode(x, dim: 256, bits: 4, method: method, seed: 1)
        let b = try q.encode(x, dim: 256, bits: 4, method: method, seed: 2)
        #expect(a.codes != b.codes)
    }

    @Test func seedPrecedence() throws {
        let x = TestVectors.uniform(count: 1, dim: 64, seed: 7)
        #expect(try q.encode(x, bits: 4, method: .turboQuant).config.seed == PITCH.defaultSeed)
        #expect(try PITCH(seed: 7).encode(x, bits: 4, method: .turboQuant).config.seed == 7)
        #expect(try q.encode(x, bits: 4, method: .turboQuant, seed: 99).config.seed == 99)
    }

    // MARK: - Invalid input

    @Test(arguments: [0, 1, 3, 100, 2048])
    func invalidDimensionThrows(dim: Int) {
        #expect(throws: PITCHError.self) {
            try q.encode([Float](repeating: 1, count: max(dim, 1)), dim: dim, bits: 4, method: .turboQuant)
        }
    }

    @Test(arguments: [0, 1, 9, 16])
    func invalidBitWidthThrows(bits: Int) {
        #expect(throws: PITCHError.invalidBitWidth(bits)) {
            try q.encode([Float](repeating: 1, count: 64), bits: bits, method: .polarQuant)
        }
    }

    @Test func emptyInputThrows() {
        #expect(throws: PITCHError.self) { try q.encode([], dim: 64, bits: 4, method: .turboQuant) }
    }

    @Test func partialVectorThrows() {
        #expect(throws: PITCHError.self) {
            try q.encode([Float](repeating: 1, count: 100), dim: 64, bits: 4, method: .turboQuant)
        }
    }

    @Test(arguments: [Float.nan, .infinity, -.infinity, 1e30, -1e18], [0, 37, 4095])
    func invalidValuesThrow(bad: Float, position: Int) {
        var x = [Float](repeating: 0.5, count: 4096); x[position] = bad
        #expect(throws: PITCHError.self) { try q.encode(x, dim: 64, bits: 4, method: .turboQuant) }
    }

    @Test(arguments: QuantMethod.allCases)
    func largestAllowedMagnitudeRoundTrips(method: QuantMethod) throws {
        let x = [Float](repeating: QuantConfig.maxInputMagnitude, count: 1024) 
        let out = try q.decode(q.encode(x, bits: 8, method: method))
        #expect(out.allSatisfy { $0.isFinite })
        #expect(ErrorMetrics.meanRelativeError(x, out, dim: 1024) < 0.01)
    }
}

//
//  RotationWorstCaseTests.swift
//  PITCHTests
//
//  The randomised Hadamard transform is not a Haar rotation: a one-hot input is
//  rotated to a flat vector (every coordinate +-1/sqrt(d)) whatever the seed, so
//  Theorem 1 does not hold for it. These tests pin the measured behaviour
//  (paper, Section 6.1) so a change to the rotation or the codebooks shows up.
//

import Foundation
import Testing
@testable import PITCH

@Suite("Rotation worst case")
struct RotationWorstCaseTests {

    static let bound2 = Float(3.0.squareRoot() * Double.pi / 2) / 16   // Theorem 1 at b = 2

    private func meanRelError(_ x: [Float], _ y: [Float], dim: Int) -> Float {
        var total: Float = 0
        for v in 0..<(x.count / dim) {
            var num: Float = 0, den: Float = 0
            for i in 0..<dim {
                let a = x[v * dim + i], d = a - y[v * dim + i]
                num += d * d; den += a * a
            }
            total += num / den
        }
        return total / Float(x.count / dim)
    }

    @Test("one-hot inputs exceed Theorem 1 (documented limitation)")
    func oneHotExceedsBound() throws {
        let pitch = try PITCH()
        let d = 64
        var x = [Float](repeating: 0, count: d * d)
        for i in 0..<d { x[i * d + i] = 1 }                       // every one-hot vector
        let batch = try pitch.encode(x, dim: d, bits: 2, method: .turboQuant)
        let err = meanRelError(x, try pitch.decode(batch), dim: d)
        #expect(abs(err - 0.25) < 1e-3, "one-hot error \(err), expected 0.2500")
        #expect(err > Self.bound2)
    }

    @Test("a single large outlier channel stays within Theorem 1")
    func outlierChannelWithinBound() throws {
        let pitch = try PITCH()
        let d = 64, n = 2048
        var rng = SystemRandomNumberGenerator()
        var x = [Float](repeating: 0, count: n * d)
        for v in 0..<n {
            for i in 0..<d {                                   // Box-Muller Gaussian
                let u1 = Float.random(in: Float.ulpOfOne..<1, using: &rng)
                let u2 = Float.random(in: 0..<1, using: &rng)
                x[v * d + i] = (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
            }
            x[v * d + Int.random(in: 0..<d, using: &rng)] *= 30
        }
        let batch = try pitch.encode(x, dim: d, bits: 2, method: .turboQuant)
        let err = meanRelError(x, try pitch.decode(batch), dim: d)
        #expect(err < Self.bound2, "outlier-channel error \(err) exceeds bound \(Self.bound2)")
    }
}

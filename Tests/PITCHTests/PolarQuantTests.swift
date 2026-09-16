import Testing
import Foundation
@testable import PITCH

@Suite("PolarQuant round-trip")
struct PolarQuantTests {

    // Shared instance
    let q = PITCH.shared

    // MARK: - Round-trip quality

    @Test func roundTrip64Bits4() throws {
        let input = randomVector(dim: 64)
        let output = try q.decode(q.encode(input, bits: 4, method: .polarQuant))
        #expect(mse(input, output) < 0.05)
    }

    @Test func roundTrip512Bits4() throws {
        let input = randomVector(dim: 512)
        let output = try q.decode(q.encode(input, bits: 4, method: .polarQuant))
        #expect(mse(input, output) < 0.05)
    }

    @Test func roundTrip512Bits8() throws {
        let input = randomVector(dim: 512)
        let output = try q.decode(q.encode(input, bits: 8, method: .polarQuant))
        #expect(mse(input, output) < 0.01)
    }

    // MARK: - Output shape and type

    @Test func outputDimensionPreserved() throws {
        let input = randomVector(dim: 256)
        let output = try q.decode(q.encode(input, bits: 4, method: .polarQuant))
        #expect(output.count == input.count)
    }

    @Test func outputIsFinite() throws {
        let input = randomVector(dim: 512)
        let output = try q.decode(q.encode(input, bits: 4, method: .polarQuant))
        #expect(output.allSatisfy { $0.isFinite })
    }

    // MARK: - Compressed envelope

    @Test func compressedMethodTag() throws {
        let c = try q.encode(randomVector(dim: 64), bits: 4, method: .polarQuant)
        #expect(c.method == .polarQuant)
    }

    @Test func packedSmallerThanOriginal() throws {
        let input = randomVector(dim: 512)
        let c = try q.encode(input, bits: 4, method: .polarQuant)
        #expect(c.packedData.count < input.count * MemoryLayout<Float>.stride)
    }

    // MARK: - Error handling

    @Test func invalidBitWidthThrows() {
        #expect(throws: PITCHError.self) {
            try q.encode(randomVector(dim: 64), bits: 5, method: .polarQuant)
        }
    }

    @Test func nonPowerOfTwoDimThrows() {
        #expect(throws: PITCHError.self) {
            try q.encode(randomVector(dim: 100), bits: 4, method: .polarQuant)
        }
    }

    @Test func emptyInputThrows() {
        #expect(throws: PITCHError.self) {
            try q.encode([], bits: 4, method: .polarQuant)
        }
    }

    // MARK: - Edge cases

    @Test func zeroVectorDoesNotCrash() throws {
        let input = [Float](repeating: 0, count: 64)
        let output = try q.decode(q.encode(input, bits: 4, method: .polarQuant))
        #expect(output.allSatisfy { $0.isFinite })
    }

    @Test func constantVectorDoesNotCrash() throws {
        let input = [Float](repeating: 3.14, count: 64)
        let output = try q.decode(q.encode(input, bits: 4, method: .polarQuant))
        #expect(output.allSatisfy { $0.isFinite })
    }
}

// MARK: - Helpers

private func randomVector(dim: Int) -> [Float] {
    (0 ..< dim).map { _ in Float.random(in: -1 ... 1) }
}

private func mse(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).map { pow($0 - $1, 2) }.reduce(0, +) / Float(a.count)
}

//
//  BatchAPITests.swift
//  PITCHTests
//
//  The batched container, the GPU-resident API, serialisation, validation and
//  storage accounting.
//

import Testing
import Foundation
import Metal
@testable import PITCH

@Suite("Batch API")
struct BatchAPITests {

    let q = PITCH.shared

    // MARK: - Batching is exact

    @Test(arguments: QuantMethod.allCases)
    func batchEqualsIndividualEncodes(method: QuantMethod) throws {
        let dim = 128, count = 16
        let x = TestVectors.gaussian(count: count, dim: dim, seed: 10)
        let batch = try q.encode(x, dim: dim, bits: 3, method: method)
        for i in 0..<count {
            let single = try q.encode(Array(x[(i * dim)..<((i + 1) * dim)]), bits: 3, method: method)
            #expect(try batch.vectors(i..<(i + 1)) == single)
        }
    }

    @Test(arguments: QuantMethod.allCases)
    func appendAndSliceRoundTrip(method: QuantMethod) throws {
        let dim = 64
        let x = TestVectors.gaussian(count: 10, dim: dim, seed: 11)
        let whole = try q.encode(x, dim: dim, bits: 4, method: method)
        var built = try whole.vectors(0..<3)
        try built.append(contentsOf: whole.vectors(3..<10))
        #expect(built == whole)
        #expect(built.count == 10)
    }

    @Test func appendRejectsDifferentConfig() throws {
        var a = try q.encode(TestVectors.gaussian(count: 1, dim: 64, seed: 12), bits: 4, method: .turboQuant)
        let b = try q.encode(TestVectors.gaussian(count: 1, dim: 64, seed: 13), bits: 3, method: .turboQuant)
        #expect(throws: PITCHError.self) { try a.append(contentsOf: b) }
    }

    // MARK: - GPU-resident API

    @Test(arguments: QuantMethod.allCases)
    func decodeIntoCallerBufferAtOffset(method: QuantMethod) throws {
        let dim = 256, count = 4, offsetFloats = 32
        let x = TestVectors.gaussian(count: count, dim: dim, seed: 14)
        let batch = try q.encode(x, dim: dim, bits: 4, method: method)
        let expected = try q.decode(batch)

        let sentinel: Float = -12345
        let buffer = makeBuffer(q.device, [Float](repeating: sentinel, count: offsetFloats + count * dim + offsetFloats))
        try q.decode(batch, into: buffer, offset: offsetFloats * 4)

        let all = readFloats(buffer, count: offsetFloats + count * dim + offsetFloats)
        #expect(Array(all[offsetFloats..<(offsetFloats + count * dim)]) == expected)
        #expect(all[..<offsetFloats].allSatisfy { $0 == sentinel })                    // untouched before
        #expect(all[(offsetFloats + count * dim)...].allSatisfy { $0 == sentinel })    // untouched after
    }

    /// Encode and decode entirely on the GPU in one caller-owned command buffer, as a
    /// Metal transformer would: nothing is copied to the CPU in between.
    @Test(arguments: QuantMethod.allCases)
    func gpuResidentPipelineMatchesArrayAPI(method: QuantMethod) throws {
        let dim = 128, count = 32
        let config = try q.config(method: method, dim: dim, bits: 4)
        let x = TestVectors.gaussian(count: count, dim: dim, seed: 15)
        let device = q.device

        let input = makeBuffer(device, x)
        let codes = try #require(device.makeBuffer(length: count * config.codeStride, options: .storageModeShared))
        let scales = try #require(device.makeBuffer(length: count * 4, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: count * dim * 4, options: .storageModeShared))

        let cb = try q.makeCommandBuffer()
        try q.enqueueEncode(input: input, codes: codes, scales: scales, count: count, config: config, commandBuffer: cb)
        try q.enqueueDecode(codes: codes, scales: scales, output: output, count: count, config: config, commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let expected = try q.decode(q.encode(x, config: config))
        #expect(readFloats(output, count: count * dim) == expected)
    }

    @Test func enqueueRejectsUndersizedBuffers() throws {
        let config = try q.config(method: .turboQuant, dim: 64, bits: 4)
        let device = q.device
        let input = makeBuffer(device, [Float](repeating: 1, count: 64 * 4))
        let codes = try #require(device.makeBuffer(length: config.codeStride * 3, options: .storageModeShared))  // needs 4
        let scales = try #require(device.makeBuffer(length: 16, options: .storageModeShared))
        let cb = try q.makeCommandBuffer()
        #expect(throws: PITCHError.self) {
            try q.enqueueEncode(input: input, codes: codes, scales: scales, count: 4, config: config, commandBuffer: cb)
        }
    }

    @Test func enqueueRejectsMisalignedOffset() throws {
        let config = try q.config(method: .polarQuant, dim: 64, bits: 4)
        let device = q.device
        let codes = try #require(device.makeBuffer(length: 1024, options: .storageModeShared))
        let scales = try #require(device.makeBuffer(length: 64, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: 4096, options: .storageModeShared))
        let cb = try q.makeCommandBuffer()
        #expect(throws: PITCHError.self) {
            try q.enqueueDecode(codes: codes, codesOffset: 2, scales: scales, output: output,
                                count: 1, config: config, commandBuffer: cb)
        }
    }

    // MARK: - Serialisation and validation

    @Test(arguments: QuantMethod.allCases)
    func codableRoundTripIsLossless(method: QuantMethod) throws {
        let x = TestVectors.gaussian(count: 5, dim: 128, seed: 16)
        let batch = try q.encode(x, dim: 128, bits: 3, method: method)
        let restored = try JSONDecoder().decode(CompressedBatch.self, from: JSONEncoder().encode(batch))
        #expect(restored == batch)
        #expect(try q.decode(restored) == q.decode(batch))
    }

    @Test func decodingRejectsWrongFormatVersion() throws {
        let batch = try q.encode(TestVectors.gaussian(count: 1, dim: 64, seed: 17), bits: 4, method: .turboQuant)
        var obj = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(batch)) as? [String: Any])
        obj["formatVersion"] = CompressedBatch.formatVersion + 1
        let tampered = try JSONSerialization.data(withJSONObject: obj)
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(CompressedBatch.self, from: tampered) }
    }

    @Test func decodingRejectsInvalidConfig() throws {
        let json = #"{"formatVersion":3,"config":{"method":"turboQuant","dim":100,"bits":4,"seed":1},"codes":"","scales":[1]}"#
        #expect(throws: (any Error).self) { try JSONDecoder().decode(CompressedBatch.self, from: Data(json.utf8)) }
    }

    @Test func constructorValidatesSizes() throws {
        let config = try q.config(method: .turboQuant, dim: 64, bits: 4)
        #expect(throws: PITCHError.self) {
            try CompressedBatch(config: config, codes: Data(count: config.codeStride - 4), scales: [1])
        }
        #expect(throws: PITCHError.self) {
            try CompressedBatch(config: config, codes: Data(count: config.codeStride), scales: [1, 2])
        }
        #expect(throws: PITCHError.self) {
            try CompressedBatch(config: config, codes: Data(count: config.codeStride), scales: [.nan])
        }
        #expect(throws: PITCHError.self) {
            try CompressedBatch(config: config, codes: Data(), scales: [])
        }
    }

    // MARK: - Storage accounting (these are the numbers the paper reports)

    @Test func perVectorAccounting() throws {
        // dim 64, 3 bits: 24 bytes of codes + 4-byte scale = 28 bytes = 3.5 bits/coordinate.
        let t = try q.config(method: .turboQuant, dim: 64, bits: 3)
        #expect(t.codeStride == 24)
        #expect(t.bytesPerVector == 28)
        #expect(t.bitsPerCoordinate == 3.5)
        // dim 1024, 4 bits: 512 + 4 bytes; 4096 / 516 = 7.94x.
        let p = try q.config(method: .polarQuant, dim: 1024, bits: 4)
        #expect(p.bytesPerVector == 516)
        #expect(abs(p.compressionRatio - 4096.0 / 516.0) < 1e-12)
        // KV caches are usually fp16: dim 64, 4 bits = 4.5 bits/coordinate = 3.56x vs fp16, 7.11x vs fp32.
        let kv = try q.config(method: .turboQuant, dim: 64, bits: 4)
        #expect(abs(kv.compressionRatioVsFloat16 - 16.0 / 4.5) < 1e-12)
        #expect(abs(kv.compressionRatio - 32.0 / 4.5) < 1e-12)
        // Packing rounds up to whole words: dim 2, 3 bits = 6 bits -> one 4-byte word.
        #expect(try q.config(method: .turboQuant, dim: 2, bits: 3).codeStride == 4)
    }

    @Test func batchAccountingIncludesHeader() throws {
        let batch = try q.encode(TestVectors.gaussian(count: 100, dim: 64, seed: 18), dim: 64, bits: 3, method: .turboQuant)
        #expect(batch.payloadByteCount == 100 * 24)
        #expect(batch.sideInformationByteCount == 100 * 4 + CompressedBatch.headerByteCount)
        #expect(batch.storedByteCount == 2_816)
        #expect(batch.originalByteCount == 25_600)
        #expect(abs(batch.bitsPerCoordinate - 2_816.0 * 8 / 6_400) < 1e-12)
        #expect(abs(batch.compressionRatioVsFloat16 - 12_800.0 / 2_816.0) < 1e-12)
    }
}

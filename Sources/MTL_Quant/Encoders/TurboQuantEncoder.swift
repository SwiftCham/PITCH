import Metal
import Foundation

final class TurboQuantEncoder {
    private let manager: DeviceManager

    init(manager: DeviceManager) {
        self.manager = manager
    }

    func encode(_ input: [Float], bits: Int) throws -> Compressed {
        return try encode(input, bits: bits, seed: UInt32.random(in: 0 ..< UInt32.max))
    }

    // Internal overload with explicit seed — used by parity tests (Phase 9)
    func encode(_ input: [Float], bits: Int, seed: UInt32) throws -> Compressed {
        let dim = input.count
        guard dim >= 2, dim <= 1024, (dim & (dim - 1)) == 0 else {
            throw MTLQuantError.invalidInput(
                "dim must be a power of two in [2, 1024], got \(dim)")
        }
        guard bits == 3 || bits == 4 || bits == 8 else {
            throw MTLQuantError.invalidBitWidth(bits)
        }

        let packedBytes   = roundUpToMultipleOf4((dim * bits + 7) / 8)
        let residualBytes = roundUpToMultipleOf4((dim + 7) / 8)

        guard
            let inputBuf    = manager.device.makeBuffer(
                bytes: input, length: dim * MemoryLayout<Float>.stride,
                options: .storageModeShared),
            let packedBuf   = manager.device.makeBuffer(
                length: packedBytes, options: .storageModeShared),
            let residualBuf = manager.device.makeBuffer(
                length: residualBytes, options: .storageModeShared),
            let metaBuf     = manager.device.makeBuffer(
                length: MemoryLayout<TurboMetaBuffer>.stride,
                options: .storageModeShared)
        else { throw MTLQuantError.encodingFailed("Could not allocate Metal buffers") }

        // Both packed and residual_bits are OR-accumulated
        memset(packedBuf.contents(),   0, packedBytes)
        memset(residualBuf.contents(), 0, residualBytes)

        var params = TurboParams(dim: UInt32(dim), bits: UInt32(bits), seed: seed, padding: 0)
        let pipeline = try manager.pipeline(named: "turbo_encode")

        guard
            let cmdBuf  = manager.commandQueue.makeCommandBuffer(),
            let encoder = cmdBuf.makeComputeCommandEncoder()
        else { throw MTLQuantError.encodingFailed("Could not create command buffer") }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuf,    offset: 0, index: 0)
        encoder.setBuffer(packedBuf,   offset: 0, index: 1)
        encoder.setBuffer(residualBuf, offset: 0, index: 2)
        encoder.setBuffer(metaBuf,     offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<TurboParams>.stride, index: 4)
        encoder.setThreadgroupMemoryLength(dim * MemoryLayout<Float>.stride, index: 0)

        // WHT requires all dim threads in exactly one threadgroup
        let tgSize = MTLSize(width: dim, height: 1, depth: 1)
        encoder.dispatchThreads(tgSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let err = cmdBuf.error {
            throw MTLQuantError.encodingFailed(err.localizedDescription)
        }

        let meta         = metaBuf.contents().load(as: TurboMetaBuffer.self)
        let packedData   = Data(bytes: packedBuf.contents(),   count: packedBytes)
        let residualData = Data(bytes: residualBuf.contents(), count: residualBytes)

        let metadata = TurboMetadata(
            bits: bits, dim: dim, seed: seed,
            scale: meta.scale, offset: meta.offset,
            residualScale: meta.residualScale,
            residualData: residualData)
        return Compressed(method: .turboQuant, packedData: packedData, metadata: metadata)
    }

    // MARK: - Decode

    func decode(_ compressed: Compressed) throws -> [Float] {
        guard let metadata = compressed.metadata as? TurboMetadata else {
            throw MTLQuantError.metadataMismatch
        }
        let dim  = metadata.dim
        let bits = metadata.bits

        var metaContents = TurboMetaBuffer(
            scale: metadata.scale, offset: metadata.offset,
            residualScale: metadata.residualScale, padding: 0)

        let packedBuf: MTLBuffer? = compressed.packedData.withUnsafeBytes { raw in
            manager.device.makeBuffer(bytes: raw.baseAddress!,
                                      length: compressed.packedData.count,
                                      options: .storageModeShared)
        }
        let residualBuf: MTLBuffer? = metadata.residualData.withUnsafeBytes { raw in
            manager.device.makeBuffer(bytes: raw.baseAddress!,
                                      length: metadata.residualData.count,
                                      options: .storageModeShared)
        }
        guard
            let packedBuf,
            let residualBuf,
            let metaBuf = manager.device.makeBuffer(
                bytes: &metaContents,
                length: MemoryLayout<TurboMetaBuffer>.stride,
                options: .storageModeShared),
            let outputBuf = manager.device.makeBuffer(
                length: dim * MemoryLayout<Float>.stride,
                options: .storageModeShared)
        else { throw MTLQuantError.encodingFailed("Could not allocate Metal buffers") }

        var params = TurboParams(dim: UInt32(dim), bits: UInt32(bits),
                                 seed: metadata.seed, padding: 0)
        let pipeline = try manager.pipeline(named: "turbo_decode")

        guard
            let cmdBuf  = manager.commandQueue.makeCommandBuffer(),
            let encoder = cmdBuf.makeComputeCommandEncoder()
        else { throw MTLQuantError.encodingFailed("Could not create command buffer") }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packedBuf,   offset: 0, index: 0)
        encoder.setBuffer(residualBuf, offset: 0, index: 1)
        encoder.setBuffer(metaBuf,     offset: 0, index: 2)
        encoder.setBuffer(outputBuf,   offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<TurboParams>.stride, index: 4)
        encoder.setThreadgroupMemoryLength(dim * MemoryLayout<Float>.stride, index: 0)

        let tgSize = MTLSize(width: dim, height: 1, depth: 1)
        encoder.dispatchThreads(tgSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let err = cmdBuf.error {
            throw MTLQuantError.encodingFailed(err.localizedDescription)
        }

        let ptr = outputBuf.contents().bindMemory(to: Float.self, capacity: dim)
        return Array(UnsafeBufferPointer(start: ptr, count: dim))
    }
}

// MARK: - Private helpers

private func roundUpToMultipleOf4(_ n: Int) -> Int {
    return (n + 3) & ~3
}

// Must match TurboParams in MTLQuantTypes.h byte-for-byte
private struct TurboParams {
    var dim:     UInt32
    var bits:    UInt32
    var seed:    UInt32
    var padding: UInt32
}

// Must match TurboMeta in MTLQuantTypes.h byte-for-byte
private struct TurboMetaBuffer {
    var scale:         Float
    var offset:        Float
    var residualScale: Float
    var padding:       Float
}

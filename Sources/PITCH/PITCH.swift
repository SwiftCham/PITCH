// The Swift Programming Language
// https://docs.swift.org/swift-book
//
//
//  PITCH.swift
//  PITCH
//
//  Public entry point.
//

import Accelerate
import Foundation
import Metal

public final class PITCH: @unchecked Sendable {

    // Default rotation seed. One seed for a whole cache gives every vector the same random rotation
    public static let defaultSeed: UInt32 = 0x5049_5443

    // Shared instance on the system default Metal device, using `defaultSeed`
    public static let shared: PITCH = {
        do { return try PITCH() }
        catch { fatalError("PITCH: could not initialise Metal: \(error.localizedDescription)") }
    }()

    // Seed used when a call doesn't pass one
    public let seed: UInt32

    ///The Metal device PITCH runs on. Buffers passed to the GPU-resident API must belong to it
    public var device: MTLDevice { manager.device }

    let manager: DeviceManager
    private let codec: Codec

    public convenience init(seed: UInt32 = PITCH.defaultSeed) throws {
        let manager = try DeviceManager()
        self.init(manager: manager, seed: seed)
    }

    // Use the `MTLDevice` your model already runs on
    public convenience init(device: MTLDevice, seed: UInt32 = PITCH.defaultSeed) throws {
        let manager = try DeviceManager(device: device)
        self.init(manager: manager, seed: seed)
    }

    private init(manager: DeviceManager, seed: UInt32) {
        self.manager = manager
        self.codec = Codec(manager: manager)
        self.seed = seed
    }

    // Configuration for this instance's seed
    public func config(method: QuantMethod, dim: Int, bits: Int) throws -> QuantConfig {
        try QuantConfig(method: method, dim: dim, bits: bits, seed: seed)
    }

    // A command buffer on PITCH's own queue, for callers who don't have one.
    public func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let cb = manager.commandQueue.makeCommandBuffer() else {
            throw PITCHError.encodingFailed("could not create command buffer")
        }
        return cb
    }

    // MARK: - Array API

    // Compresses `vectors.count / dim` vectors stored contiguously.
    public func encode(_ vectors: [Float], dim: Int, bits: Int, method: QuantMethod,
                       seed: UInt32? = nil) throws -> CompressedBatch {
        let config = try QuantConfig(method: method, dim: dim, bits: bits, seed: seed ?? self.seed)
        return try encode(vectors, config: config)
    }

    // Compresses one vector (`dim` = `vector.count`).
    public func encode(_ vector: [Float], bits: Int, method: QuantMethod,
                       seed: UInt32? = nil) throws -> CompressedBatch {
        try encode(vector, dim: vector.count, bits: bits, method: method, seed: seed)
    }

    public func encode(_ vectors: [Float], config: QuantConfig) throws -> CompressedBatch {
        guard !vectors.isEmpty, vectors.count % config.dim == 0 else {
            throw PITCHError.invalidInput("\(vectors.count) floats is not a whole number of \(config.dim)-dim vectors")
        }
        try Self.validateValues(vectors)
        let count = vectors.count / config.dim
        let device = manager.device

        guard let input = vectors.withUnsafeBytes({
                  device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let codes = device.makeBuffer(length: count * config.codeStride, options: .storageModeShared),
              let scales = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)
        else { throw PITCHError.encodingFailed("could not allocate Metal buffers") }

        let cb = try makeCommandBuffer()
        try codec.encodeCompress(input: input, inputOffset: 0, codes: codes, codesOffset: 0,
                                 scales: scales, scalesOffset: 0,
                                 count: count, config: config, commandBuffer: cb)
        try run(cb, failure: PITCHError.encodingFailed)

        let codeData = Data(bytes: codes.contents(), count: count * config.codeStride)
        let scaleArray = Array(UnsafeBufferPointer(
            start: scales.contents().bindMemory(to: Float.self, capacity: count), count: count))
        return CompressedBatch(unchecked: config, codes: codeData, scales: scaleArray)
    }

    // Reconstructs every vector in the batch, contiguously.
    public func decode(_ batch: CompressedBatch) throws -> [Float] {
        let n = batch.count * batch.config.dim
        guard let output = manager.device.makeBuffer(length: n * MemoryLayout<Float>.size,
                                                     options: .storageModeShared) else {
            throw PITCHError.decodingFailed("could not allocate Metal buffers")
        }
        try decode(batch, into: output, offset: 0)
        return Array(UnsafeBufferPointer(start: output.contents().bindMemory(to: Float.self, capacity: n), count: n))
    }

    // Reconstructs the batch into `buffer` starting at byte `offset`, then waits.
    public func decode(_ batch: CompressedBatch, into buffer: MTLBuffer, offset: Int = 0) throws {
        try batch.validate()
        let device = manager.device
        guard let codes = batch.codes.withUnsafeBytes({
                  device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }),
              let scales = batch.scales.withUnsafeBytes({
                  device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        else { throw PITCHError.decodingFailed("could not allocate Metal buffers") }

        let cb = try makeCommandBuffer()
        try codec.encodeDecompress(codes: codes, codesOffset: 0, scales: scales, scalesOffset: 0,
                                   output: buffer, outputOffset: offset,
                                   count: batch.count, config: batch.config, commandBuffer: cb)
        try run(cb, failure: PITCHError.decodingFailed)
    }

    // MARK: - GPU-resident API

    /// Encodes compression of `count` contiguous float32 vectors into `commandBuffer`.
    ///
    /// Writes `count * config.codeStride` bytes of codes and `count` float32 scales.
    /// the results are valid once `commandBuffer` completes.
    /// All buffers must belong to `device`, and offsets must be multiples of 4.
    ///
    /// The input is not inspected on this path: values must be finite with magnitude at most
    /// `QuantConfig.maxInputMagnitude`, or the kernel's float32 sum of squares overflows.
    public func enqueueEncode(input: MTLBuffer, inputOffset: Int = 0,
                              codes: MTLBuffer, codesOffset: Int = 0,
                              scales: MTLBuffer, scalesOffset: Int = 0,
                              count: Int, config: QuantConfig,
                              commandBuffer: MTLCommandBuffer) throws {
        try codec.encodeCompress(input: input, inputOffset: inputOffset,
                                 codes: codes, codesOffset: codesOffset,
                                 scales: scales, scalesOffset: scalesOffset,
                                 count: count, config: config, commandBuffer: commandBuffer)
    }

    // Encodes decompression of `count` vectors into `commandBuffer`
    public func enqueueDecode(codes: MTLBuffer, codesOffset: Int = 0,
                              scales: MTLBuffer, scalesOffset: Int = 0,
                              output: MTLBuffer, outputOffset: Int = 0,
                              count: Int, config: QuantConfig,
                              commandBuffer: MTLCommandBuffer) throws {
        try codec.encodeDecompress(codes: codes, codesOffset: codesOffset,
                                   scales: scales, scalesOffset: scalesOffset,
                                   output: output, outputOffset: outputOffset,
                                   count: count, config: config, commandBuffer: commandBuffer)
    }

    // MARK: - Private

    // Rejects NaN, infinity, and magnitudes the kernels cannot square without overflow.
    // Vectorised with Accelerate
    // TODO: accelerate removal to open codebase to cross langauge compat
    static func validateValues(_ x: [Float]) throws {
        guard vDSP.sum(vDSP.multiply(Float(0), x)).isFinite else {
            throw PITCHError.invalidInput("input contains NaN or infinity")
        }
        let largest = vDSP.maximumMagnitude(x)
        guard largest <= QuantConfig.maxInputMagnitude else {
            throw PITCHError.invalidInput(
                "input magnitude \(largest) exceeds \(QuantConfig.maxInputMagnitude); the kernels' float32 sum of squares would overflow")
        }
    }

    private func run(_ cb: MTLCommandBuffer, failure: (String) -> PITCHError) throws {
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw failure(error.localizedDescription) }
        guard cb.status == .completed else { throw failure("command buffer status \(cb.status.rawValue)") }
    }
}

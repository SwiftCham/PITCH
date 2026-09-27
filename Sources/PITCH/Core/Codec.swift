//
//  Codec.swift
//  PITCH
//
//  Encodes PITCH's compute commands into a command buffer. 
//
//  Created by Benjamin Stacey on 23/09/2026.

import Metal

// Mirrors `struct BatchParams` in PITCHKernels.metal.
struct BatchParams {
    var dim: UInt32
    var bits: UInt32
    var seed: UInt32
    var count: UInt32
}

struct Codec {
    let manager: DeviceManager

    init(manager: DeviceManager) {
        precondition(MemoryLayout<BatchParams>.size == 16 && MemoryLayout<BatchParams>.stride == 16,
                     "BatchParams layout must match PITCHKernels.metal (16 bytes)")
        self.manager = manager
    }

    private func params(_ config: QuantConfig, count: Int) throws -> BatchParams {
        guard count >= 1 else { throw PITCHError.invalidInput("count must be at least 1") }
        guard count * config.dim <= Int(UInt32.max) else {
            throw PITCHError.invalidInput("batch too large: count * dim must fit in 32 bits")
        }
        return BatchParams(dim: UInt32(config.dim), bits: UInt32(config.bits),
                           seed: config.seed, count: UInt32(count))
    }

    private func check(_ buffer: MTLBuffer, offset: Int, needs bytes: Int, name: String) throws {
        guard buffer.device === manager.device else {
            throw PITCHError.invalidInput("\(name) buffer belongs to a different MTLDevice")
        }
        guard offset >= 0, offset % 4 == 0 else {
            throw PITCHError.invalidInput("\(name) offset \(offset) must be a non-negative multiple of 4")
        }
        guard buffer.length >= offset + bytes else {
            throw PITCHError.invalidInput("\(name) buffer has \(buffer.length - offset) bytes after offset; needs \(bytes)")
        }
    }

    private func pipeline(_ name: String, dim: Int) throws -> MTLComputePipelineState {
        let p = try manager.pipeline(named: name)
        guard dim <= p.maxTotalThreadsPerThreadgroup else {
            throw PITCHError.invalidDimension(dim)   // thrown when gpu pipeline is unsupported
        }
        return p
    }

    private func threadgroupBytes(_ dim: Int) -> Int {
        (dim * MemoryLayout<Float>.stride + 15) & ~15      // must be a multiple of 16
    }

    private func codebook(_ config: QuantConfig) throws -> [Float] {
        guard let cb = TurboCodebooks.codebook(dim: config.dim, bits: config.bits) else {
            throw PITCHError.invalidInput("no TurboQuant codebook for dim \(config.dim), bits \(config.bits)")
        }
        return cb
    }

    // MARK: - Encode

    func encodeCompress(input: MTLBuffer, inputOffset: Int,
                        codes: MTLBuffer, codesOffset: Int,
                        scales: MTLBuffer, scalesOffset: Int,
                        count: Int, config: QuantConfig,
                        commandBuffer: MTLCommandBuffer) throws {
        var p = try params(config, count: count)
        try check(input, offset: inputOffset, needs: count * config.dim * 4, name: "input")
        try check(codes, offset: codesOffset, needs: count * config.codeStride, name: "codes")
        try check(scales, offset: scalesOffset, needs: count * 4, name: "scales")

        let kernel = config.method == .turboQuant ? "turbo_encode" : "polar_encode"
        let pipeline = try self.pipeline(kernel, dim: config.dim)
        let cb: [Float] = try config.method == .turboQuant ? codebook(config) : []
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PITCHError.encodingFailed("could not create compute encoder")
        }
        enc.label = "PITCH \(kernel)"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: inputOffset, index: 0)
        enc.setBuffer(codes, offset: codesOffset, index: 1)
        enc.setBuffer(scales, offset: scalesOffset, index: 2)
        switch config.method {
        case .turboQuant:
            cb.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 3) }
            enc.setBytes(&p, length: MemoryLayout<BatchParams>.stride, index: 4)
        case .polarQuant:
            enc.setBytes(&p, length: MemoryLayout<BatchParams>.stride, index: 3)
        }
        enc.setThreadgroupMemoryLength(threadgroupBytes(config.dim), index: 0)   // scratch floats
        enc.setThreadgroupMemoryLength(threadgroupBytes(config.dim), index: 1)
        enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: config.dim, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Decode

    func encodeDecompress(codes: MTLBuffer, codesOffset: Int,
                          scales: MTLBuffer, scalesOffset: Int,
                          output: MTLBuffer, outputOffset: Int,
                          count: Int, config: QuantConfig,
                          commandBuffer: MTLCommandBuffer) throws {
        var p = try params(config, count: count)
        try check(codes, offset: codesOffset, needs: count * config.codeStride, name: "codes")
        try check(scales, offset: scalesOffset, needs: count * 4, name: "scales")
        try check(output, offset: outputOffset, needs: count * config.dim * 4, name: "output")

        let kernel = config.method == .turboQuant ? "turbo_decode" : "polar_decode"
        let pipeline = try self.pipeline(kernel, dim: config.dim)
        let cb: [Float] = try config.method == .turboQuant ? codebook(config) : []
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PITCHError.decodingFailed("could not create compute encoder")
        }
        enc.label = "PITCH \(kernel)"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(codes, offset: codesOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        switch config.method {
        case .turboQuant:
            cb.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
            enc.setBuffer(output, offset: outputOffset, index: 3)
            enc.setBytes(&p, length: MemoryLayout<BatchParams>.stride, index: 4)
        case .polarQuant:
            enc.setBuffer(output, offset: outputOffset, index: 2)
            enc.setBytes(&p, length: MemoryLayout<BatchParams>.stride, index: 3)
        }
        enc.setThreadgroupMemoryLength(threadgroupBytes(config.dim), index: 0)
        enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: config.dim, height: 1, depth: 1))
        enc.endEncoding()
    }
}

//
//  DeviceManager.swift
//  PITCH
//
//  Created by Benjamin Stacey on 24/06/2026.
//

import Metal
import Foundation

// Owns the Metal device, command queue, compiled library and pipeline cache.
final class DeviceManager: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private var library: MTLLibrary?
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    static let shaderFileName = "PITCHKernels"

    convenience init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PITCHError.metalNotSupported
        }
        try self.init(device: device)
    }

    init(device: MTLDevice) throws {
        guard let queue = device.makeCommandQueue() else {
            throw PITCHError.metalNotSupported
        }
        self.device = device
        self.commandQueue = queue
    }

    // Compiled pipeline for a kernel, created once and cached
    func pipeline(named name: String) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }

        if let cached = pipelineCache[name] { return cached }
        let lib = try loadLibraryLocked()
        guard let function = lib.makeFunction(name: name) else {
            throw PITCHError.shaderCompilationFailed("function \(name) not found in \(Self.shaderFileName).metal")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelineCache[name] = pipeline
        return pipeline
    }

    private func loadLibraryLocked() throws -> MTLLibrary {
        if let library { return library }

        if let lib = try? device.makeDefaultLibrary(bundle: .module),
           lib.functionNames.contains("turbo_encode") {
            library = lib
            return lib
        }
        guard let url = Bundle.module.url(forResource: Self.shaderFileName, withExtension: "metal"),
              let source = try? String(contentsOf: url, encoding: .utf8) else {
            throw PITCHError.shaderCompilationFailed("could not read \(Self.shaderFileName).metal from the bundle")
        }
        do {
            let lib = try device.makeLibrary(source: source, options: nil)
            library = lib
            return lib
        } catch {
            throw PITCHError.shaderCompilationFailed("\(Self.shaderFileName).metal: \(error.localizedDescription)")
        }
    }
}

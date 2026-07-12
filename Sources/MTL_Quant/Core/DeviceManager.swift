//
//  DeviceManager.swift
//  MTL_Quant
//
//  Created by Benjamin Stacey on 24/06/2026.
//
import Metal
import Foundation

final class DeviceManager: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let cacheLock = NSLock()

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MTLQuantError.metalNotSupported
        }
        guard let queue = device.makeCommandQueue() else {
            throw MTLQuantError.encodingFailed("Could not create command queue")
        }
        self.device = device
        self.commandQueue = queue
    }

    init(device: MTLDevice) throws {
        guard let queue = device.makeCommandQueue() else {
            throw MTLQuantError.encodingFailed("Could not create command queue")
        }
        self.device = device
        self.commandQueue = queue
    }

    func pipeline(named name: String) throws -> MTLComputePipelineState {
        cacheLock.lock()
        defer { cacheLock.unlock() }

        if let cached = pipelineCache[name] {
            return cached
        }

        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            throw MTLQuantError.shaderCompilationFailed("Could not load default Metal library")
        }
        guard let function = library.makeFunction(name: name) else {
            throw MTLQuantError.shaderCompilationFailed("Function not found: \(name)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelineCache[name] = pipeline
        return pipeline
    }
}

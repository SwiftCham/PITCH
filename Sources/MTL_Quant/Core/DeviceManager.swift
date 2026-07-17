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

        if let cached = pipelineCache[name] { return cached }

        let library = try loadLibrary(forKernel: name)
        guard let function = library.makeFunction(name: name) else {
            throw MTLQuantError.shaderCompilationFailed("Function not found: \(name)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelineCache[name] = pipeline
        return pipeline
    }

    // Metal struct definitions which are substituted for #include "MTLQuantTypes.h" during runtime compilation
    private static let metalTypesSource = """
        typedef struct { uint dim; uint bits; uint seed; uint padding; } TurboParams;
        typedef struct { float scale; float offset; float residualScale; float padding; } TurboMeta;
        typedef struct { uint dim; uint bits; uint seed; uint padding; } PolarParams;
        typedef struct { float magnitude; float padding[3]; } PolarMeta;
        """

    // Kernel function maps
    private let kernelToFile: [String: String] = [
        "turbo_encode": "TurboEncoder",
        "turbo_decode": "TurboDecoder",
        "polar_encode": "PolarEncoder",
        "polar_decode": "PolarDecoder"
    ]

    private func loadLibrary(forKernel name: String) throws -> MTLLibrary {
        if let lib = try? device.makeDefaultLibrary(bundle: .module) {
            return lib
        }
        guard let fileName = kernelToFile[name] else {
            throw MTLQuantError.shaderCompilationFailed("No source file registered for kernel: \(name)")
        }
        guard let metalURL = Bundle.module.url(forResource: fileName, withExtension: "metal"),
              var source = try? String(contentsOf: metalURL, encoding: .utf8) else {
            throw MTLQuantError.shaderCompilationFailed("Could not read \(fileName).metal from bundle")
        }

        // Runtime Metal compilation has no access to system C headers (no stdint.h)
        source = source.replacingOccurrences(
            of: "#include \"MTLQuantTypes.h\"",
            with: Self.metalTypesSource
        )

        do {
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MTLQuantError.shaderCompilationFailed("\(fileName).metal: \(error.localizedDescription)")
        }
    }
}

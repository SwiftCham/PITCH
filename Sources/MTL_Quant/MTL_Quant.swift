// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import Metal

public final class MTL_Quant: @unchecked Sendable {

    // shared mtldevice singleton lazily init
    public static let shared: MTL_Quant = {
        try! MTL_Quant()
    }()

    let deviceManager: DeviceManager
    private let polarEncoder: PolarQuantEncoder

    public init() throws {
        let dm = try DeviceManager()
        self.deviceManager  = dm
        self.polarEncoder   = PolarQuantEncoder(manager: dm)
    }

    // mtldevice injection for already setup projects
    public init(device: MTLDevice) throws {
        let dm = try DeviceManager(device: device)
        self.deviceManager  = dm
        self.polarEncoder   = PolarQuantEncoder(manager: dm)
    }

    // MARK: - Encode

    public func encode(_ tensor: [Float], bits: Int, method: Method) throws -> Compressed {
        switch method {
        case .polarQuant:
            return try polarEncoder.encode(tensor, bits: bits)
        case .turboQuant:
            throw MTLQuantError.encodingFailed("TurboQuant encoder not yet implemented")
        }
    }

    // MARK: - Decode

    public func decode(_ compressed: Compressed) throws -> [Float] {
        switch compressed.method {
        case .polarQuant:
            return try polarEncoder.decode(compressed)
        case .turboQuant:
            throw MTLQuantError.encodingFailed("TurboQuant decoder not yet implemented")
        }
    }
}

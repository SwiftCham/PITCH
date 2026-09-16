// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import Metal

public final class PITCH: @unchecked Sendable {

    // shared mtldevice singleton lazily init
    public static let shared: PITCH = {
        try! PITCH()
    }()

    let deviceManager: DeviceManager
    private let turboEncoder: TurboQuantEncoder
    private let polarEncoder: PolarQuantEncoder

    public init() throws {
        let dm = try DeviceManager()
        self.deviceManager = dm
        self.turboEncoder  = TurboQuantEncoder(manager: dm)
        self.polarEncoder  = PolarQuantEncoder(manager: dm)
    }

    // mtldevice injection for already setup projects
    public init(device: MTLDevice) throws {
        let dm = try DeviceManager(device: device)
        self.deviceManager = dm
        self.turboEncoder  = TurboQuantEncoder(manager: dm)
        self.polarEncoder  = PolarQuantEncoder(manager: dm)
    }

    // MARK: - Encode

    public func encode(_ tensor: [Float], bits: Int, method: QuantMethod) throws -> Compressed {
        switch method {
        case .turboQuant: return try turboEncoder.encode(tensor, bits: bits)
        case .polarQuant: return try polarEncoder.encode(tensor, bits: bits)
        }
    }

    // MARK: - Decode

    public func decode(_ compressed: Compressed) throws -> [Float] {
        switch compressed.method {
        case .turboQuant: return try turboEncoder.decode(compressed)
        case .polarQuant: return try polarEncoder.decode(compressed)
        }
    }
}

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

    public init() throws {
        self.deviceManager = try DeviceManager()
    }

    // mtldevice injection for already setup projects
    public init(device: MTLDevice) throws {
        self.deviceManager = try DeviceManager(device: device)
    }
}

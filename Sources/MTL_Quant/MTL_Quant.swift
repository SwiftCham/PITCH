// The Swift Programming Language
// https://docs.swift.org/swift-book


import Foundation
import Metal

public class MTL_Quant {
    
    // instance init
    private let deviceManager: DeviceManager
    
    // device init
    public init(device: MTLDevice? = nil) {
        self.deviceManager = DeviceManager(device: device)
    }
}
 

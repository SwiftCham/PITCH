//
//  DeviceManager.swift
//  MTL_Quant
//
//  Created by Benjamin Stacey on 24/06/2026.
//
import Metal

class DeviceManager {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    init(device: MTLDevice? = nil) {
        self.device = device ?? MTLCreateSystemDefaultDevice()!
        self.commandQueue = self.device.makeCommandQueue()!
    }
}

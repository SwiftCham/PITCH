//
//  Compressed.swift
//  PITCH
//
//  Created by Benjamin Stacey on 10/07/2026.
//

import Foundation

public struct Compressed: Sendable {
    public let method: QuantMethod
    public let packedData: Data
    public let metadata: any Metadata

    public init(method: QuantMethod, packedData: Data, metadata: any Metadata) {
        self.method = method
        self.packedData = packedData
        self.metadata = metadata
    }
}

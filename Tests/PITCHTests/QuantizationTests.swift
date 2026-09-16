import Testing
import Foundation
import Metal
import PITCH

@Suite("Quantization Tests")
struct QuantizationTests {
    
    @Test("Basic 8-bits quantization should reduce size and preserve fidelity")
    func test8BitsQuantization() async throws {
        let quantizer = try PITCH.shared
        let weights: [Float] = Array(repeating: 1.0, count: 32) // Model weights
        
        // Original size in bytes: 32 floats * 4 bytes/float = 128 bytes
        let compressed = try quantizer.encode(
            weights,
            bits: 8,
            method: .turboQuant
        )
        
        // 8-bit compression should reduce size to 32 bytes (32 weights * 1 byte/weight)
        #expect(compressed.packedData.count == 32, "Expected 8-bit compression to reduce to 32 bytes")
        
        // Decompress and validate the results
        let decompressed = try quantizer.decode(compressed)
        #expect(decompressed.count == weights.count, "Decompressed array should match original size of 32 weights")
        
        // Ensure data fidelity
        let maxAllowedError = 0.001 // You can adjust this threshold based on requirements
        let mse = Double(decompressed.reduce(Float(0)) { $0 + pow($1 - 1.0, 2) }) / Double(decompressed.count)
        #expect(mse < maxAllowedError, "MSE should be below \(maxAllowedError)")
    }
    
    @Test("4-bits quantization should further reduce size with acceptable fidelity")
    func test4BitsQuantization() async throws {
        let quantizer = try PITCH.shared
        let weights: [Float] = Array(repeating: 1.0, count: 32)
        
        // Original size in bytes: 128 bytes
        let compressed = try quantizer.encode(
            weights,
            bits: 4,
            method: .turboQuant
        )
        
        // 4-bit compression should reduce to 16 bytes (32 weights * 0.5 bytes/weight)
        #expect(compressed.packedData.count == 16, "Expected 4-bit compression to reduce to 16 bytes")
        
        let decompressed = try quantizer.decode(compressed)
        #expect(decompressed.count == weights.count, "Decompressed array should match original size of 32 weights")
        
        // Allow higher error tolerance for lower bit depths
        let maxAllowedError = 0.1
        let mse = Double(decompressed.reduce(Float(0)) { $0 + pow($1 - 1.0, 2) }) / Double(decompressed.count)
        #expect(mse < maxAllowedError, "MSE should be below \(maxAllowedError)")
    }
    
    @Test("3-bit quantization (minimum supported precision) stays within an acceptable error bound")
    func minimumBitWidthAccuracyBound() async throws {
        let quantizer = try PITCH.shared
        let weights: [Float] = Array(repeating: 1.0, count: 32)
        
        // 3-bit packing for 32 weights: ceil(32*3/8) = 12 bytes
        let compressed = try quantizer.encode(weights, bits: 3, method: .turboQuant)
        #expect(compressed.packedData.count == 12, "Expected 3-bit compression to pack to 12 bytes")
        
        let decompressed = try quantizer.decode(compressed)
        #expect(decompressed.count == weights.count, "Decompressed array should match original size of 32 weights")
        
        let maxAllowedError = 1.0
        let mse = Double(decompressed.reduce(Float(0)) { $0 + pow($1 - 1.0, 2) }) / Double(decompressed.count)
        #expect(mse < maxAllowedError, "MSE should be below \(maxAllowedError)")
    }
}

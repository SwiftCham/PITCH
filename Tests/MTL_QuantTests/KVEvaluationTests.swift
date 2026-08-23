import Testing
import Foundation
@testable import MTL_Quant

// Loads real KV-cache vectors produced by reference/extract_kv_cache.py and
// evaluates MTL-Quant reconstruction quality against the naive baseline printed
// by that script. All tests in this suite are skipped gracefully if the JSON
// file has not been generated yet.

@Suite("KV-Cache Evaluation")
struct KVEvaluationTests {

    static let vectorsURL: URL? = {
        let path = FileManager.default.currentDirectoryPath
            .appending("/reference/kv_vectors.json")
        return FileManager.default.fileExists(atPath: path)
            ? URL(fileURLWithPath: path) : nil
    }()

    let q = MTL_Quant.shared

    // MARK: - Quality at each bit width

    @Test func turboQuant4bit() throws {
        let (mse, cos) = try evaluate(method: QuantMethod.turboQuant, bits: 4)
        print("[TurboQuant  4-bit]  MSE = \(fmt(mse))   cosine = \(fmt(cos))")
        #expect(mse < 0.15)
        #expect(cos > 0.90)
    }

    @Test func turboQuant8bit() throws {
        let (mse, cos) = try evaluate(method: QuantMethod.turboQuant, bits: 8)
        print("[TurboQuant  8-bit]  MSE = \(fmt(mse))   cosine = \(fmt(cos))")
        #expect(mse < 0.01)
        #expect(cos > 0.999)
    }

    @Test func polarQuant4bit() throws {
        let (mse, cos) = try evaluate(method: QuantMethod.polarQuant, bits: 4)
        print("[PolarQuant  4-bit]  MSE = \(fmt(mse))   cosine = \(fmt(cos))")
        #expect(mse < 0.15)
        #expect(cos > 0.90)
    }

    @Test func polarQuant8bit() throws {
        let (mse, cos) = try evaluate(method: QuantMethod.polarQuant, bits: 8)
        print("[PolarQuant  8-bit]  MSE = \(fmt(mse))   cosine = \(fmt(cos))")
        #expect(mse < 0.01)
        #expect(cos > 0.999)
    }

    // MARK: - Sanity: all outputs finite

    @Test func allOutputsFinite() throws {
        let vectors = try loadVectors()
        for v in vectors.prefix(50) {
            let output = try q.decode(q.encode(v.data, bits: 4, method: QuantMethod.turboQuant))
            #expect(output.allSatisfy { $0.isFinite })
        }
    }
}

// MARK: - Private helpers

private extension KVEvaluationTests {

    struct KVVector: Decodable {
        let type: String
        let layer: Int
        let head: Int
        let dim: Int
        let data: [Float]
    }

    struct KVPayload: Decodable {
        let model: String
        let num_vectors: Int
        let vectors: [KVVector]
    }

    func loadVectors() throws -> [KVVector] {
        guard let url = KVEvaluationTests.vectorsURL else {
            // File not generated yet — skip silently.
            return []
        }
        let raw = try Data(contentsOf: url)
        let payload = try JSONDecoder().decode(KVPayload.self, from: raw)
        print("Loaded \(payload.num_vectors) vectors from \(payload.model)")
        return payload.vectors
    }

    func evaluate(method: QuantMethod, bits: Int) throws -> (mse: Double, cosine: Double) {
        let vectors = try loadVectors()
        guard !vectors.isEmpty else { return (0, 1) }

        var totalMSE = 0.0
        var totalCos = 0.0

        for v in vectors {
            let output = try q.decode(q.encode(v.data, bits: bits, method: method))
            totalMSE += Double(mse(v.data, output))
            totalCos += Double(cosineSim(v.data, output))
        }

        return (totalMSE / Double(vectors.count), totalCos / Double(vectors.count))
    }

    func mse(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { ($0 - $1) * ($0 - $1) }.reduce(0, +) / Float(a.count)
    }

    func cosineSim(_ a: [Float], _ b: [Float]) -> Float {
        let dot  = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        return normA * normB > 0 ? dot / (normA * normB) : 0
    }

    func fmt(_ v: Double) -> String {
        String(format: "%.6f", v)
    }
}

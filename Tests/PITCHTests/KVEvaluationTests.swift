//
//  KVEvaluationTests.swift
//  PITCHTests
//
//  Reconstruction quality on real KV-cache vectors from reference/extract_kv_cache.py.
//  Looks for reference/kv_vectors*.json (or the directory in PITCH_KV_DIR); t

import Testing
import Foundation
@testable import PITCH

// MARK: - Data loading (shared with BenchmarkSuite)

struct KVSet: Sendable {
    let model: String
    let kind: String
    let dim: Int
    let data: [Float]
    var count: Int { data.count / dim }
    var label: String { "\(model) \(kind)s" }
}

let kvFiles: [URL] = {
    let dir = ProcessInfo.processInfo.environment["PITCH_KV_DIR"]
        ?? FileManager.default.currentDirectoryPath + "/reference"
    let names = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
    return names.filter { $0.hasPrefix("kv_vectors") && $0.hasSuffix(".json") }
        .sorted()
        .map { URL(fileURLWithPath: dir).appendingPathComponent($0) }
}()

func loadKVSets() throws -> [KVSet] {
    struct Vector: Decodable { let type: String; let dim: Int; let data: [Float] }
    struct Payload: Decodable { let model: String; let vectors: [Vector] }
    var sets: [KVSet] = []
    for url in kvFiles {
        let payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        for kind in ["key", "value"] {
            let vs = payload.vectors.filter { $0.type == kind }
            guard let dim = vs.first?.dim, vs.allSatisfy({ $0.dim == dim }) else { continue }
            sets.append(KVSet(model: payload.model, kind: kind, dim: dim, data: vs.flatMap(\.data)))
        }
    }
    return sets
}

func naiveRoundToNearest(_ x: [Float], dim: Int, bits: Int) -> [Float] {
    let levels = Float((1 << bits) - 1)
    var out = x
    for v in 0..<(x.count / dim) {
        let r = (v * dim)..<((v + 1) * dim)
        guard let lo = x[r].min(), let hi = x[r].max(), hi > lo else { continue }
        let step = (hi - lo) / levels
        for i in r { out[i] = min(max(((x[i] - lo) / step).rounded(), 0), levels) * step + lo }
    }
    return out
}

// MARK: - Suite

@Suite("KV-cache evaluation",
       .enabled(if: !kvFiles.isEmpty, "no reference/kv_vectors*.json found; run reference/extract_kv_cache.py"))
struct KVEvaluationTests {

    let q = PITCH.shared

    struct Case: Sendable, CustomTestStringConvertible {
        let method: QuantMethod
        let bits: Int
        let maxError: Double
        let minCosine: Double
        var testDescription: String { "\(method.rawValue) \(bits)-bit" }
    }

    // Reference values on GPT-2 / Qwen2.5 head_dim 64 (rate_distortion.py), ~1.5x margin.
    static let cases: [Case] = [
        Case(method: .turboQuant, bits: 3, maxError: 0.05,  minCosine: 0.97),
        Case(method: .turboQuant, bits: 4, maxError: 0.014, minCosine: 0.99),
        Case(method: .turboQuant, bits: 8, maxError: 1e-4,  minCosine: 0.9999),
        Case(method: .polarQuant, bits: 3, maxError: 0.08,  minCosine: 0.96),
        Case(method: .polarQuant, bits: 4, maxError: 0.021, minCosine: 0.985),
        Case(method: .polarQuant, bits: 8, maxError: 8e-5,  minCosine: 0.9999),
    ]

    @Test(arguments: KVEvaluationTests.cases)
    func reconstructionQuality(_ c: Case) throws {
        for set in try loadKVSets() {
            let batch = try q.encode(set.data, dim: set.dim, bits: c.bits, method: c.method)
            let out = try q.decode(batch)
            let e = ErrorMetrics.meanRelativeError(set.data, out, dim: set.dim)
            let cos = ErrorMetrics.meanCosine(set.data, out, dim: set.dim)
            let naive = ErrorMetrics.meanRelativeError(
                set.data, naiveRoundToNearest(set.data, dim: set.dim, bits: c.bits), dim: set.dim)
            print("[\(set.label)] \(c.testDescription): n=\(set.count) "
                  + String(format: "bpc=%.3f relErr=%.3e cos=%.6f | naive %ld-bit (bpc=%.3f) relErr=%.3e",
                           batch.config.bitsPerCoordinate, e, cos, c.bits,
                           Double(c.bits) + 64.0 / Double(set.dim), naive))
            #expect(e < c.maxError, "\(set.label): relative error \(e)")
            #expect(cos > c.minCosine, "\(set.label): cosine \(cos)")
        }
    }
    
    @Test(arguments: [2, 3, 4])
    func turboQuantBeatsNaiveRoundingWithFewerBits(bits: Int) throws {
        for set in try loadKVSets() {
            let out = try q.decode(q.encode(set.data, dim: set.dim, bits: bits, method: .turboQuant))
            let e = ErrorMetrics.meanRelativeError(set.data, out, dim: set.dim)
            let naive = ErrorMetrics.meanRelativeError(
                set.data, naiveRoundToNearest(set.data, dim: set.dim, bits: bits), dim: set.dim)
            #expect(e < naive, "\(set.label) \(bits)-bit: TurboQuant \(e) vs naive \(naive)")
        }
    }

    @Test func allOutputsFinite() throws {
        for set in try loadKVSets() {
            for method in QuantMethod.allCases {
                let out = try q.decode(q.encode(set.data, dim: set.dim, bits: 2, method: method))
                #expect(out.allSatisfy { $0.isFinite }, "\(set.label) \(method.rawValue)")
            }
        }
    }
}

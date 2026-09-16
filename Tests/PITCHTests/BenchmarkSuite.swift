//
//  BenchmarkSuite.swift
//  PITCH
//  generates MSE, cosine similarity, inner-product distortion,
//  true compression ratio (packed data + metadata overhead), and
//  encode/decode throughput (GB/s) this test will always pass as it is made sheerly for benchmarking

// TODO: REMOVE IN PROD

import Testing
import Foundation
@testable import PITCH

@Suite("Benchmark Suite - Thesis Results")
struct BenchmarkSuite {
    static let bitWidths = [3, 4, 8]
    static let dims = [64, 128, 256, 512, 1024]
    static let trialsPerCell = 30
    static let timingRepeats = 20

    let q = PITCH.shared

    @Test func runFullBenchmark() throws {
        let pool = try loadVectorPool()
        var results: [BenchmarkResult] = []

        for method in [QuantMethod.turboQuant, QuantMethod.polarQuant] {
            for dim in Self.dims {
                for bits in Self.bitWidths {
                    let result = try benchmarkCell(method: method, bits: bits, dim: dim, pool: pool)
                    results.append(result)
                    print(format(result))
                }
            }
        }

        try writeCSV(results, to: "benchmark_results.csv")
        #expect(!results.isEmpty)
    }

    private func benchmarkCell(method: QuantMethod, bits: Int, dim: Int, pool: [[Float]]) throws -> BenchmarkResult {
        var totalMSE: Double = 0
        var totalIPError: Double = 0
        var totalCompressedBytes: Int = 0
        var encodeTimes: [Double] = []
        var decodeTimes: [Double] = []

        let warmupVec = vector(forDim: dim, index: 0, pool: pool)
        _ = try q.decode(q.encode(warmupVec, bits: bits, method: method))

        for trial in 0..<Self.trialsPerCell {
            let x = vector(forDim: dim, index: trial, pool: pool)
            let y = randomGaussianVector(dim: dim, seed: UInt32(truncatingIfNeeded: 9_973 &* trial &+ 17))

            let (compressed, encodeTime) = try timed { try q.encode(x, bits: bits, method: method) }
            let (decoded, decodeTime) = try timed { try q.decode(compressed) }

            totalMSE += Double(mse(x, decoded))
            totalIPError += Double(innerProductSquaredError(x, decoded, y))
            totalCompressedBytes += compressedSize(compressed)

            if trial < Self.timingRepeats {
                encodeTimes.append(encodeTime)
                decodeTimes.append(decodeTime)
            }
        }

        let n = Double(Self.trialsPerCell)
        let avgMSE = Float(totalMSE / n)
        let avgIPError = Float(totalIPError / n)
        let avgCompressedBytes = Double(totalCompressedBytes) / n
        let originalBytes = Double(dim * MemoryLayout<Float>.stride)
        let compressionRatio = Float(originalBytes / avgCompressedBytes)

        let encodeThroughput = throughputGBs(bytes: originalBytes, times: encodeTimes)
        let decodeThroughput = throughputGBs(bytes: originalBytes, times: decodeTimes)

        return BenchmarkResult(
            algorithm: method,
            bits: bits,
            dim: dim,
            mse: avgMSE,
            innerProductError: avgIPError,
            encodeThroughputGBs: encodeThroughput,
            decodeThroughputGBs: decodeThroughput,
            compressionRatio: compressionRatio
        )
    }

    private func mse(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { ($0 - $1) * ($0 - $1) }.reduce(0, +) / Float(a.count)
    }

    private func innerProductSquaredError(_ x: [Float], _ xHat: [Float], _ y: [Float]) -> Float {
        let exact = zip(x, y).map(*).reduce(0, +)
        let approx = zip(xHat, y).map(*).reduce(0, +)
        return (exact - approx) * (exact - approx)
    }

    private func compressedSize(_ c: Compressed) -> Int {
        switch c.metadata {
        case let t as TurboMetadata:
            return c.packedData.count + t.residualData.count + MemoryLayout<Float>.size * 3
        case is PolarMetadata:
            return c.packedData.count + MemoryLayout<Float>.size
        default:
            return c.packedData.count
        }
    }

    private func timed<T>(_ block: () throws -> T) throws -> (T, Double) {
        let start = DispatchTime.now()
        let result = try block()
        let end = DispatchTime.now()
        let seconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
        return (result, seconds)
    }

    private func throughputGBs(bytes: Double, times: [Double]) -> Float {
        guard !times.isEmpty else { return 0 }
        let avgTime = times.reduce(0, +) / Double(times.count)
        guard avgTime > 0 else { return 0 }
        return Float((bytes / avgTime) / 1e9)
    }

    private func loadVectorPool() throws -> [[Float]] {
        let path = FileManager.default.currentDirectoryPath.appending("/reference/kv_vectors.json")
        guard FileManager.default.fileExists(atPath: path) else {
            print("[BenchmarkSuite] reference/kv_vectors.json not found - using synthetic Gaussian vectors")
            return []
        }
        struct KVVector: Decodable { let data: [Float] }
        struct KVPayload: Decodable { let vectors: [KVVector] }
        let raw = try Data(contentsOf: URL(fileURLWithPath: path))
        let payload = try JSONDecoder().decode(KVPayload.self, from: raw)
        print("[BenchmarkSuite] Loaded \(payload.vectors.count) real KV vectors from \(path)")
        return payload.vectors.map { $0.data }
    }

    private func vector(forDim dim: Int, index: Int, pool: [[Float]]) -> [Float] {
        let candidates = pool.filter { $0.count == dim }
        if !candidates.isEmpty {
            return candidates[index % candidates.count]
        }
        return randomGaussianVector(dim: dim, seed: UInt32(truncatingIfNeeded: 1_000_003 &* index &+ dim))
    }

    private func randomGaussianVector(dim: Int, seed: UInt32) -> [Float] {
        var generator = SeededGenerator(seed: seed)
        var result = [Float](repeating: 0, count: dim)
        var i = 0
        while i < dim {
            let u1 = Float.random(in: 1e-9...1, using: &generator)
            let u2 = Float.random(in: 0...1, using: &generator)
            let r = sqrt(-2 * log(u1))
            let theta = 2 * Float.pi * u2
            result[i] = r * cos(theta)
            if i + 1 < dim { result[i + 1] = r * sin(theta) }
            i += 2
        }
        return result
    }

    private func format(_ r: BenchmarkResult) -> String {
        let alg = r.algorithm.rawValue.padding(toLength: 12, withPad: " ", startingAt: 0)
        return "[\(alg)] dim=\(String(format: "%4d", r.dim))  bits=\(r.bits)  " +
               "MSE=\(String(format: "%.6f", r.mse))  " +
               "IPErr=\(String(format: "%.6f", r.innerProductError))  " +
               "ratio=\(String(format: "%.2fx", r.compressionRatio))  " +
               "enc=\(String(format: "%.2f", r.encodeThroughputGBs))GB/s  " +
               "dec=\(String(format: "%.2f", r.decodeThroughputGBs))GB/s"
    }

    private func writeCSV(_ results: [BenchmarkResult], to filename: String) throws {
        var lines = ["algorithm,bits,dim,mse,inner_product_error,compression_ratio,encode_gbs,decode_gbs"]
        for r in results {
            lines.append([
                r.algorithm.rawValue,
                "\(r.bits)",
                "\(r.dim)",
                "\(r.mse)",
                "\(r.innerProductError)",
                "\(r.compressionRatio)",
                "\(r.encodeThroughputGBs)",
                "\(r.decodeThroughputGBs)"
            ].joined(separator: ","))
        }
        let csv = lines.joined(separator: "\n")
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(filename)
        try csv.write(to: url, atomically: true, encoding: .utf8)
        print("[BenchmarkSuite] Wrote \(results.count) rows to \(url.path)")
    }
}

struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt32) { state = UInt64(seed) &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

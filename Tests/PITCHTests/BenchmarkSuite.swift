//
//  BenchmarkSuite.swift
//  PITCHTests
//
//  Generates the paper's data. Not a test, so it is SKIPPED unless enabled:
//
//      PITCH_RUN_BENCHMARKS=1 swift test --scratch-path /tmp/pitch-build --filter BenchmarkSuite
//
//  Writes to ./results/:
//    environment.txt  device, OS, date
//    accuracy.csv     error vs bits on Gaussian and real KV vectors
//    throughput.csv   per-call GPU time (command-buffer timestamps) and wall time,
//                     for the GPU-resident API and the array API, across batch sizes

import Testing
import Foundation
import Metal
@testable import PITCH

private let benchmarksEnabled = ProcessInfo.processInfo.environment["PITCH_RUN_BENCHMARKS"] == "1"

@Suite("Benchmark Suite", .serialized,
       .enabled(if: benchmarksEnabled, "set PITCH_RUN_BENCHMARKS=1 to generate results/*.csv"))
struct BenchmarkSuite {

    let q = PITCH.shared
    static let bitWidths = Array(QuantConfig.supportedBits)
    static let dims = [64, 128, 256, 512, 1024]
    static let batchSizes = [1, 16, 256, 4096]
    static let timingRepeats = 100
    static let sustainedDispatches = 16
    static let sustainedRepeats = 20
    static let warmUpSeconds = 0.25
    static let arrayRepeats = 30

    private var resultsDir: URL {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("results")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - Environment

    @Test func recordEnvironment() throws {
        let text = """
        device: \(q.device.name)
        os: \(ProcessInfo.processInfo.operatingSystemVersionString)
        date: \(ISO8601DateFormatter().string(from: Date()))
        max threads per threadgroup: \(q.device.maxThreadsPerThreadgroup.width)
        """
        try text.write(to: resultsDir.appendingPathComponent("environment.txt"), atomically: true, encoding: .utf8)
        print(text)
    }

    // MARK: - Accuracy

    @Test func accuracySweep() throws {
        var rows = ["source,kind,method,bits,dim,count,bits_per_coord,ratio_vs_fp32,ratio_vs_fp16,rel_err,cosine,ip_rel_rmse,ip_slope"]

        var sets: [KVSet] = Self.dims.map { d in
            KVSet(model: "gaussian", kind: "synthetic", dim: d,
                  data: TestVectors.gaussian(count: 2048, dim: d, seed: UInt64(d)))
        }
        sets += try loadKVSets()

        for set in sets {
            let queries = TestVectors.gaussian(count: set.count, dim: set.dim, seed: 99)
            for method in QuantMethod.allCases {
                for bits in Self.bitWidths {
                    let batch = try q.encode(set.data, dim: set.dim, bits: bits, method: method)
                    let out = try q.decode(batch)
                    let (rmse, slope) = innerProductError(set.data, out, queries, dim: set.dim)
                    rows.append([set.model, set.kind, method.rawValue, "\(bits)", "\(set.dim)", "\(set.count)",
                                 fmt(batch.config.bitsPerCoordinate),
                                 fmt(batch.config.compressionRatio), fmt(batch.config.compressionRatioVsFloat16),
                                 fmt(ErrorMetrics.meanRelativeError(set.data, out, dim: set.dim)),
                                 fmt(ErrorMetrics.meanCosine(set.data, out, dim: set.dim)),
                                 fmt(rmse), fmt(slope)].joined(separator: ","))
                }
            }
            print("[accuracy] \(set.label) d=\(set.dim) done")
        }
        try write(rows, "accuracy.csv")
    }
    private func innerProductError(_ x: [Float], _ xHat: [Float], _ y: [Float], dim: Int) -> (Double, Double) {
        var se = 0.0, cross = 0.0, ref = 0.0
        let n = x.count / dim
        for v in 0..<n {
            var exact = 0.0, approx = 0.0
            for i in (v * dim)..<((v + 1) * dim) {
                exact += Double(x[i]) * Double(y[i]); approx += Double(xHat[i]) * Double(y[i])
            }
            se += (approx - exact) * (approx - exact); cross += approx * exact; ref += exact * exact
        }
        return ((se / ref).squareRoot(), cross / ref)
    }

    // MARK: - Throughput

    @Test func throughputSweep() throws {
        var rows = ["method,bits,dim,batch,phase,api,gpu_us_median,gpu_us_p10,gpu_us_p90,wall_us_median,wall_us_p10,wall_us_p90,"
                    + "sustained_gpu_us_per_call,sustained_wall_us_per_call,ns_per_vector_sustained_gpu,input_gb_per_s_sustained_gpu"]
        let device = q.device

        for dim in Self.dims {
            for batch in Self.batchSizes {
                let x = TestVectors.gaussian(count: batch, dim: dim, seed: 7)   // generated once per shape
                let input = makeBuffer(device, x)
                let output = try #require(device.makeBuffer(length: batch * dim * 4, options: .storageModeShared))
                let bytes = Double(batch * dim * 4)

                for method in QuantMethod.allCases {
                    for bits in [2, 4, 8] {
                        let config = try q.config(method: method, dim: dim, bits: bits)
                        let codes = try #require(device.makeBuffer(length: batch * config.codeStride, options: .storageModeShared))
                        let scales = try #require(device.makeBuffer(length: batch * 4, options: .storageModeShared))

                        // GPU-resident API: pre-allocated buffers, one command buffer per call.
                        let enc = try timeCalls {
                            try q.enqueueEncode(input: input, codes: codes, scales: scales,
                                                count: batch, config: config, commandBuffer: $0)
                        }
                        let dec = try timeCalls {
                            try q.enqueueDecode(codes: codes, scales: scales, output: output,
                                                count: batch, config: config, commandBuffer: $0)
                        }
                        let compressed = try q.encode(x, config: config)
                        let arrayEnc = try wallTime { _ = try q.encode(x, config: config) }
                        let arrayDec = try wallTime { _ = try q.decode(compressed) }

                        rows.append(row(method, bits, dim, batch, "encode", "gpu_resident", enc, bytes))
                        rows.append(row(method, bits, dim, batch, "decode", "gpu_resident", dec, bytes))
                        rows.append(row(method, bits, dim, batch, "encode", "array", Timing(wall: arrayEnc), bytes))
                        rows.append(row(method, bits, dim, batch, "decode", "array", Timing(wall: arrayDec), bytes))
                    }
                }
            }
            print("[throughput] d=\(dim) done")
        }
        try write(rows, "throughput.csv")
    }

    private struct Timing {                               // all in seconds
        var gpu: [Double] = []
        var wall: [Double] = []
        var sustainedGPU: [Double] = []                    // per dispatch
        var sustainedWall: [Double] = []                   // per dispatch
    }

    private func timeCalls(_ encode: (MTLCommandBuffer) throws -> Void) throws -> Timing {
        var t = Timing()

        // Sustained warm-up: back-to-back work until the GPU clock has settled :)
        let warmUntil = DispatchTime.now().uptimeNanoseconds + UInt64(Self.warmUpSeconds * 1e9)
        repeat {
            let cb = try q.makeCommandBuffer()
            for _ in 0..<Self.sustainedDispatches { try encode(cb) }
            cb.commit(); cb.waitUntilCompleted()
        } while DispatchTime.now().uptimeNanoseconds < warmUntil

        for _ in 0..<Self.timingRepeats {
            let cb = try q.makeCommandBuffer()
            try encode(cb)
            let start = DispatchTime.now().uptimeNanoseconds
            cb.commit(); cb.waitUntilCompleted()
            let end = DispatchTime.now().uptimeNanoseconds
            guard cb.status == .completed else { throw PITCHError.encodingFailed("command buffer failed") }
            t.gpu.append(cb.gpuEndTime - cb.gpuStartTime)
            t.wall.append(Double(end - start) / 1e9)
        }

        // sustained load
        let k = Double(Self.sustainedDispatches)
        for _ in 0..<Self.sustainedRepeats {
            let start = DispatchTime.now().uptimeNanoseconds
            let cb = try q.makeCommandBuffer()
            for _ in 0..<Self.sustainedDispatches { try encode(cb) }
            cb.commit(); cb.waitUntilCompleted()
            let end = DispatchTime.now().uptimeNanoseconds
            guard cb.status == .completed else { throw PITCHError.encodingFailed("command buffer failed") }
            t.sustainedGPU.append((cb.gpuEndTime - cb.gpuStartTime) / k)
            t.sustainedWall.append(Double(end - start) / 1e9 / k)
        }
        return t
    }

    private func wallTime(_ block: () throws -> Void) throws -> [Double] {
        for _ in 0..<3 { try block() }                                          // warm-up
        var out: [Double] = []
        for _ in 0..<Self.arrayRepeats {
            let start = DispatchTime.now().uptimeNanoseconds
            try block()
            out.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
        }
        return out
    }

    private func row(_ m: QuantMethod, _ bits: Int, _ dim: Int, _ batch: Int, _ phase: String, _ api: String,
                     _ t: Timing, _ bytes: Double) -> String {
        func us(_ v: [Double], _ p: Double) -> String { v.isEmpty ? "" : fmt(percentile(v, p) * 1e6) }
        let sGPU: Double? = t.sustainedGPU.isEmpty ? nil : percentile(t.sustainedGPU, 0.5)
        return [m.rawValue, "\(bits)", "\(dim)", "\(batch)", phase, api,
                us(t.gpu, 0.5), us(t.gpu, 0.1), us(t.gpu, 0.9),
                us(t.wall, 0.5), us(t.wall, 0.1), us(t.wall, 0.9),
                us(t.sustainedGPU, 0.5), us(t.sustainedWall, 0.5),
                sGPU.map { fmt($0 * 1e9 / Double(batch)) } ?? "",
                sGPU.map { fmt(bytes / $0 / 1e9) } ?? ""].joined(separator: ",")
    }

    private func percentile(_ v: [Double], _ p: Double) -> Double {
        guard !v.isEmpty else { return .nan }
        let s = v.sorted()
        return s[min(s.count - 1, Int((Double(s.count - 1) * p).rounded()))]
    }

    private func fmt(_ v: Double) -> String { String(format: "%.6g", v) }

    private func write(_ rows: [String], _ name: String) throws {
        let url = resultsDir.appendingPathComponent(name)
        try rows.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        print("[benchmark] wrote \(rows.count - 1) rows to \(url.path)")
    }
}

//
//  PerChannelBenchmark.swift
//  PITCHTests
//
//  Timing for the per-channel key mode. Skipped unless enabled:
//
//      PITCH_RUN_BENCHMARKS=1 swift test -c release --scratch-path /tmp/pitch-build --filter PerChannelBenchmark
//
//  Writes results/per_channel_throughput.csv. TurboQuant is timed on the same shapes in the
//  same run, so the two are compared under identical GPU conditions.
//

import Testing
import Foundation
import Metal
@testable import PITCH

private let benchmarksEnabled = ProcessInfo.processInfo.environment["PITCH_RUN_BENCHMARKS"] == "1"

@Suite("Per-channel benchmark", .serialized,
       .enabled(if: benchmarksEnabled, "set PITCH_RUN_BENCHMARKS=1 to generate results/per_channel_throughput.csv"))
struct PerChannelBenchmark {

    let q = PITCH.shared
    static let dims = [64, 128, 256, 512, 1024]
    static let tokenCounts = [64, 256, 4096]
    static let bitWidths = [2, 4, 8]
    static let timingRepeats = 100
    static let sustainedDispatches = 16
    static let sustainedRepeats = 20
    static let warmUpSeconds = 0.25

    @Test func perChannelThroughput() throws {
        var rows = ["method,bits,dim,tokens,phase,gpu_us_median,wall_us_median,"
                    + "sustained_gpu_us_per_call,sustained_wall_us_per_call,ns_per_token_sustained_gpu"]
        let device = q.device

        for dim in Self.dims {
            for tokens in Self.tokenCounts {
                let x = TestVectors.gaussian(count: tokens, dim: dim, seed: 7)
                let input = makeBuffer(device, x)
                let output = try #require(device.makeBuffer(length: tokens * dim * 4, options: .storageModeShared))

                for bits in Self.bitWidths {
                    // Per-channel
                    let cc = try q.channelConfig(dim: dim, bits: bits)
                    let codes = try #require(device.makeBuffer(length: tokens * cc.codeStride, options: .storageModeShared))
                    let ranges = try #require(device.makeBuffer(
                        length: cc.groupCount(tokens: tokens) * cc.rangeStride, options: .storageModeShared))
                    let enc = try timeCalls {
                        try q.enqueuePerChannelEncode(input: input, codes: codes, ranges: ranges,
                                                      count: tokens, config: cc, commandBuffer: $0)
                    }
                    let dec = try timeCalls {
                        try q.enqueuePerChannelDecode(codes: codes, ranges: ranges, output: output,
                                                      count: tokens, config: cc, commandBuffer: $0)
                    }
                    rows.append(row("perChannel", bits, dim, tokens, "encode", enc))
                    rows.append(row("perChannel", bits, dim, tokens, "decode", dec))

                    // TurboQuant, same shapes, same run
                    let tc = try q.config(method: .turboQuant, dim: dim, bits: bits)
                    let tCodes = try #require(device.makeBuffer(length: tokens * tc.codeStride, options: .storageModeShared))
                    let tNorms = try #require(device.makeBuffer(length: tokens * 4, options: .storageModeShared))
                    let tEnc = try timeCalls {
                        try q.enqueueEncode(input: input, codes: tCodes, scales: tNorms,
                                            count: tokens, config: tc, commandBuffer: $0)
                    }
                    let tDec = try timeCalls {
                        try q.enqueueDecode(codes: tCodes, scales: tNorms, output: output,
                                            count: tokens, config: tc, commandBuffer: $0)
                    }
                    rows.append(row("turboQuant", bits, dim, tokens, "encode", tEnc))
                    rows.append(row("turboQuant", bits, dim, tokens, "decode", tDec))
                }
            }
            print("[per-channel benchmark] d=\(dim) done")
        }

        let dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("results")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("per_channel_throughput.csv")
        try rows.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        print("[per-channel benchmark] wrote \(rows.count - 1) rows to \(url.path)")
    }

    private struct Timing { var gpu: [Double] = []; var wall: [Double] = []; var sGPU: [Double] = []; var sWall: [Double] = [] }

    private func timeCalls(_ encode: (MTLCommandBuffer) throws -> Void) throws -> Timing {
        var t = Timing()
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
        let k = Double(Self.sustainedDispatches)
        for _ in 0..<Self.sustainedRepeats {
            let start = DispatchTime.now().uptimeNanoseconds
            let cb = try q.makeCommandBuffer()
            for _ in 0..<Self.sustainedDispatches { try encode(cb) }
            cb.commit(); cb.waitUntilCompleted()
            let end = DispatchTime.now().uptimeNanoseconds
            guard cb.status == .completed else { throw PITCHError.encodingFailed("command buffer failed") }
            t.sGPU.append((cb.gpuEndTime - cb.gpuStartTime) / k)
            t.sWall.append(Double(end - start) / 1e9 / k)
        }
        return t
    }

    private func row(_ m: String, _ bits: Int, _ dim: Int, _ tokens: Int, _ phase: String, _ t: Timing) -> String {
        let sGPU = median(t.sGPU)
        return [m, "\(bits)", "\(dim)", "\(tokens)", phase,
                fmt(median(t.gpu) * 1e6), fmt(median(t.wall) * 1e6),
                fmt(sGPU * 1e6), fmt(median(t.sWall) * 1e6),
                fmt(sGPU * 1e9 / Double(tokens))].joined(separator: ",")
    }

    private func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
    private func fmt(_ v: Double) -> String { String(format: "%.6g", v) }
}

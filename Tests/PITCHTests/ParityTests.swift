//
//  ParityTests.swift
//  PITCHTests
//
//  Checks the Metal kernels against reference/pitch_reference.py, via fixtures written by
//  reference/gen_parity_fixture.py.
//
//

import Testing
import Foundation
@testable import PITCH

private struct ParityFixture: Decodable {
    struct Group: Decodable { let name: String; let dim: Int; let count: Int; let input: String }
    struct Case: Decodable {
        let group: String
        let method: QuantMethod
        let bits: Int
        let seed: UInt32
        let codes: String
        let scales: String
        let reconstruction: String
        let relative_error: Double
    }
    let groups: [Group]
    let cases: [Case]

    static func load() throws -> ParityFixture {
        let url = Bundle.module.url(forResource: "parity", withExtension: "json", subdirectory: "Fixtures")
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("Fixtures/parity.json")
        return try JSONDecoder().decode(ParityFixture.self, from: Data(contentsOf: url))
    }
}

@Suite("Parity with reference implementation")
struct ParityTests {

    let q = PITCH.shared

    @Test func fixtureIsPresentAndComplete() throws {
        let fixture = try ParityFixture.load()
        #expect(fixture.cases.count == 40)
        #expect(Set(fixture.cases.map(\.method)) == Set(QuantMethod.allCases))
    }

    @Test(arguments: QuantMethod.allCases)
    func metalMatchesReference(method: QuantMethod) throws {
        let fixture = try ParityFixture.load()
        let groups = Dictionary(uniqueKeysWithValues: fixture.groups.map { ($0.name, $0) })

        for c in fixture.cases where c.method == method {
            let g = try #require(groups[c.group])
            let label = "\(c.group) \(c.method.rawValue) \(c.bits)-bit"
            let config = try QuantConfig(method: c.method, dim: g.dim, bits: c.bits, seed: c.seed)
            let input = floats(base64: g.input)
            let refCodes = try #require(Data(base64Encoded: c.codes))
            let refScales = floats(base64: c.scales)
            let refRecon = floats(base64: c.reconstruction)

            // 1 + 2: encode
            let ours = try q.encode(input, config: config)
            let a = unpackCodes(ours.codes, count: g.count, dim: g.dim, bits: c.bits)
            let b = unpackCodes(refCodes, count: g.count, dim: g.dim, bits: c.bits)
            let mismatches = zip(a, b).filter { $0 != $1 }.count
            #expect(mismatches <= max(2, a.count / 500), "\(label): \(mismatches) of \(a.count) codes differ")
            for (s, r) in zip(ours.scales, refScales) {
                #expect(abs(s - r) <= 1e-5 * max(1, abs(r)), "\(label): scale \(s) vs \(r)")
            }

            // 3: decode the reference's codes
            let refBatch = try CompressedBatch(config: config, codes: refCodes, scales: refScales)
            let decoded = try q.decode(refBatch)
            let diff = ErrorMetrics.maxRelativeDifference(decoded, refRecon, dim: g.dim)
            #expect(diff < 1e-5, "\(label): decode differs from reference by \(diff)")

            // 4: end-to-end error
            let e = ErrorMetrics.meanRelativeError(input, try q.decode(ours), dim: g.dim)
            #expect(abs(e - c.relative_error) <= 0.02 * c.relative_error + 1e-7,
                    "\(label): error \(e) vs reference \(c.relative_error)")
        }
    }
}

// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PITCH",
    products: [
        .library(
            name: "PITCH",
            targets: ["PITCH"]
        ),
    ],
    targets: [
        .target(
            name: "PITCH",
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "PITCHTests",
            dependencies: ["PITCH"],
            resources: [
                .copy("Fixtures"),
                ]
        ),
    ],
    swiftLanguageModes: [.v6]
)

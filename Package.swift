// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MTL_Quant",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MTL_Quant",
            targets: ["MTL_Quant"]
        ),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MTL_Quant",
            resources: [
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "MTL_QuantTests",
            dependencies: ["MTL_Quant"]
        ),
    ],
    swiftLanguageModes: [.v6]
)

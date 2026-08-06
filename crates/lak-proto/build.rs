//! Build script for Protocol Buffers code generation
//!
//! Uses tonic-build (which wraps prost-build) to compile .proto files
//! into Rust types and gRPC service stubs.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &["proto/lak.proto"],
            &["proto/"], // include path for google.protobuf imports
        )?;

    // Re-run if proto files change
    println!("cargo:rerun-if-changed=proto/lak.proto");

    Ok(())
}

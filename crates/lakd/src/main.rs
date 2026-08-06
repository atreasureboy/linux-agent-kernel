//! LAKd — Linux Agent Kernel Daemon
//!
//! The main daemon process that boots the Agent Kernel.
//! Entry point for the entire LAK system.
//!
//! Phase 1: Userspace prototype running as a gRPC server.

mod server;

use std::net::SocketAddr;
use std::sync::Arc;

use lak_services::kernel::KernelService;
use lak_core::traits::AgentKernel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("lak=debug".parse()?)
                .add_directive("tonic=info".parse()?),
        )
        .init();

    tracing::info!("[LAK] Linux Agent Kernel Daemon starting...");
    tracing::info!("[LAK] Version: {}", env!("CARGO_PKG_VERSION"));

    // Print banner
    print_banner();

    tracing::info!("[LAK] Initializing Agent Kernel...");

    // Create the kernel service — the heart of LAK
    let kernel: Arc<dyn AgentKernel> = Arc::new(KernelService::new());

    tracing::info!("[LAK] Kernel initialized. Building gRPC server...");

    // Create the gRPC server bridge
    let grpc_service = server::LakGrpcServer::new(Arc::clone(&kernel));

    // Bind to the configured address
    let addr: SocketAddr = "0.0.0.0:9191".parse()?;

    tracing::info!("[LAK] gRPC server listening on {addr}");
    tracing::info!("[LAK] Ready. Accepting agent connections.");

    // Start the gRPC server with graceful shutdown
    tonic::transport::Server::builder()
        .add_service(grpc_service.into_service())
        .serve_with_shutdown(addr, async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to listen for CTRL+C");
            tracing::info!("[LAK] Received shutdown signal, draining...");
        })
        .await?;

    tracing::info!("[LAK] Shutdown complete.");

    Ok(())
}

fn print_banner() {
    eprintln!(
        r#"
╔══════════════════════════════════════════════════════════╗
║     Linux Agent Kernel (LAK) — 智能体内核               ║
║     Version 0.1.0 — Phase 1 MVP                         ║
║                                                          ║
║     "Not an OS for humans. An OS for agents."            ║
╚══════════════════════════════════════════════════════════╝
"#
    );
}

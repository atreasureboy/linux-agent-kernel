//! LAKd — Linux Agent Kernel Daemon
//!
//! The main daemon process that boots the Agent Kernel.
//! Entry point for the entire LAK system.
//!
//! Phase 1: Userspace prototype running as a gRPC server.
//!
//! Configuration (environment variables):
//! - `LAK_LISTEN_ADDR`   — gRPC bind address (default `0.0.0.0:9191`)
//! - `LAK_MAX_AGENTS`    — maximum number of concurrent agents (default 1000)
//! - `OPENAI_API_KEY`    — enables the OpenAI driver (model via `OPENAI_MODEL`)
//! - `ANTHROPIC_API_KEY` — enables the Anthropic driver (model via `ANTHROPIC_MODEL`)
//! - `OLLAMA_URL`        — enables the Ollama driver (default when set: model via `OLLAMA_MODEL`)
//! - `LAK_DISABLE_CLOUD_LLM` — when set to `1`, skip cloud drivers even if keys exist

mod server;

use std::net::SocketAddr;
use std::sync::Arc;

use lak_core::traits::AgentKernel;
use lak_services::kernel::KernelService;

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
    let max_agents: u32 = std::env::var("LAK_MAX_AGENTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let kernel = Arc::new(KernelService::new().with_max_agents(max_agents));

    // Register LLM drivers from the environment
    register_llm_drivers(&kernel).await;

    tracing::info!("[LAK] Kernel initialized. Building gRPC server...");

    // Create the gRPC server bridge (upcast to the trait object)
    let kernel_trait: Arc<dyn AgentKernel> = Arc::clone(&kernel) as Arc<dyn AgentKernel>;
    let grpc_service = server::LakGrpcServer::new(kernel_trait);

    // Bind to the configured address
    let addr: SocketAddr = std::env::var("LAK_LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9191".into())
        .parse()?;

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

/// Register every LLM driver configured through environment variables.
async fn register_llm_drivers(kernel: &KernelService) {
    let disable_cloud = std::env::var("LAK_DISABLE_CLOUD_LLM").as_deref() == Ok("1");

    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        if !key.is_empty() && !disable_cloud {
            let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".into());
            let driver = lak_tal::llm::openai::OpenAIDriver::new(key, model);
            kernel.add_driver(Arc::new(driver)).await;
            tracing::info!("[LAK] Registered OpenAI driver");
        }
    }

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        if !key.is_empty() && !disable_cloud {
            let model =
                std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-5".into());
            let driver = lak_tal::llm::anthropic::AnthropicDriver::new(key, model);
            kernel.add_driver(Arc::new(driver)).await;
            tracing::info!("[LAK] Registered Anthropic driver");
        }
    }

    if let Ok(url) = std::env::var("OLLAMA_URL") {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.1".into());
        let driver = lak_tal::llm::ollama::OllamaDriver::new(model).with_base_url(url);
        kernel.add_driver(Arc::new(driver)).await;
        tracing::info!("[LAK] Registered Ollama driver");
    }
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

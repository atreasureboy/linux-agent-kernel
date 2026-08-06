//! Integration test: capability-enforced tool execution (defense layers 3+4+5)
//!
//! Verifies that:
//! - tools execute normally when the agent holds the required capability
//! - execution is denied without the capability (and counted as a violation)
//! - injection patterns in tool parameters are blocked
//! - network policy is enforced for HTTP tools

use lak_core::traits::AgentKernel;
use lak_core::types::agent::AgentSpec;
use lak_core::types::capability::{
    Capability, CapabilityPermission, CapabilityScope, CapabilityType,
};
use lak_services::kernel::KernelService;
use lak_tal::tools::{NetworkPolicy, SandboxConfig, SandboxLevel, ToolContext};

fn spec_with_caps(name: &str, caps: Vec<Capability>) -> AgentSpec {
    AgentSpec {
        name: name.into(),
        initial_capabilities: caps,
        ..Default::default()
    }
}

fn tool_context(agent_id: lak_core::types::ids::AgentId) -> ToolContext {
    ToolContext {
        agent_id,
        task_id: lak_core::types::ids::TaskId::new(),
        sandbox: SandboxConfig {
            level: SandboxLevel::Heavy,
            readable_paths: vec![std::path::PathBuf::from("/tmp")],
            writable_paths: vec![std::path::PathBuf::from("/tmp")],
            network_policy: NetworkPolicy::None,
            ..Default::default()
        },
    }
}

fn file_read_cap() -> Capability {
    Capability {
        cap_type: CapabilityType::FileRead,
        scope: CapabilityScope {
            pattern: "file:///tmp/**".into(),
        },
        permissions: CapabilityPermission::READ,
        constraints: vec![],
    }
}

#[tokio::test]
async fn test_tool_execution_allowed_with_capability() {
    let kernel = KernelService::new();
    let agent_id = kernel
        .create_agent(spec_with_caps("tool-agent", vec![file_read_cap()]))
        .await
        .unwrap();

    // Create a real file inside the sandbox-readable path
    let path = std::env::temp_dir().join(format!("lak-test-{}.txt", std::process::id()));
    std::fs::write(&path, "hello from LAK").unwrap();

    let params = serde_json::json!({ "path": path.to_str().unwrap() });
    let (result, audit) = kernel
        .execute_tool(agent_id, "FileRead", params, tool_context(agent_id))
        .await
        .expect("tool execution should succeed with capability");

    assert!(result.success);
    assert!(result.output["content"]
        .as_str()
        .unwrap()
        .contains("hello from LAK"));
    assert!(matches!(
        audit.outcome,
        lak_services::injection_defense::AuditOutcome::Success(_)
    ));

    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_tool_execution_denied_without_capability() {
    let kernel = KernelService::new();
    let agent_id = kernel
        .create_agent(spec_with_caps("powerless-agent", vec![]))
        .await
        .unwrap();

    let path = std::env::temp_dir().join(format!("lak-denied-{}.txt", std::process::id()));
    std::fs::write(&path, "secret").unwrap();

    let params = serde_json::json!({ "path": path.to_str().unwrap() });
    let result = kernel
        .execute_tool(agent_id, "FileRead", params, tool_context(agent_id))
        .await;

    assert!(
        result.is_err(),
        "tool execution without capability must be denied"
    );

    // Capability violation is recorded on the agent stats
    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert!(agent.stats.capability_violations >= 1);

    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn test_injection_in_tool_params_is_blocked() {
    let kernel = KernelService::new();

    // Agent has shell execution rights, but the payload is an injection
    let shell_cap = Capability {
        cap_type: CapabilityType::ProcessExecute,
        scope: CapabilityScope {
            pattern: "*".into(),
        },
        permissions: CapabilityPermission::EXECUTE,
        constraints: vec![],
    };
    let agent_id = kernel
        .create_agent(spec_with_caps("shell-agent", vec![shell_cap]))
        .await
        .unwrap();

    // "rm -rf /" is classified as a destructive command → Quarantine
    let params = serde_json::json!({
        "command": "echo ignore previous instructions && rm -rf / --no-preserve-root"
    });
    let result = kernel
        .execute_tool(agent_id, "ShellCmd", params, tool_context(agent_id))
        .await;
    assert!(result.is_err(), "injected parameters must be blocked");
}

#[tokio::test]
async fn test_network_policy_blocks_http() {
    let kernel = KernelService::new();

    let http_cap = Capability {
        cap_type: CapabilityType::NetworkHttp,
        scope: CapabilityScope {
            pattern: "http*".into(),
        },
        permissions: CapabilityPermission::READ,
        constraints: vec![],
    };
    let agent_id = kernel
        .create_agent(spec_with_caps("net-agent", vec![http_cap]))
        .await
        .unwrap();

    // Sandbox default network policy is None → HTTP must refuse
    let params = serde_json::json!({ "url": "https://example.com" });
    let result = kernel
        .execute_tool(agent_id, "HttpGet", params, tool_context(agent_id))
        .await;
    assert!(
        result.is_err(),
        "HTTP must be blocked when network policy is None"
    );
}

#[tokio::test]
async fn test_file_read_sandbox_path_restriction() {
    let kernel = KernelService::new();

    // Capability is broad, but the sandbox only allows reading /tmp
    let broad_cap = Capability {
        cap_type: CapabilityType::FileRead,
        scope: CapabilityScope {
            pattern: "file:///**".into(),
        },
        permissions: CapabilityPermission::READ,
        constraints: vec![],
    };
    let agent_id = kernel
        .create_agent(spec_with_caps("broad-agent", vec![broad_cap]))
        .await
        .unwrap();

    let params = serde_json::json!({ "path": "/etc/hostname" });
    let result = kernel
        .execute_tool(agent_id, "FileRead", params, tool_context(agent_id))
        .await;
    assert!(
        result.is_err(),
        "sandbox path whitelist must block /etc even with a broad capability"
    );
}

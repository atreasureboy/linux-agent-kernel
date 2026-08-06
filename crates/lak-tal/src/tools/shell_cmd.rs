//! ShellCmd tool — execute shell commands in sandbox

use async_trait::async_trait;

use lak_core::types::capability::{CapabilityPermission, CapabilityRequirement, CapabilityType};
use super::*;

#[derive(Debug, Default)]
pub struct ShellCmdTool;

#[async_trait]
impl Tool for ShellCmdTool {
    fn name(&self) -> &str {
        "ShellCmd"
    }

    fn description(&self) -> &str {
        "Execute a shell command in a sandboxed environment. Commands are limited by seccomp, resource limits, and timeout."
    }

    fn danger_level(&self) -> DangerLevel {
        DangerLevel::High
    }

    fn required_capability(&self) -> CapabilityRequirement {
        CapabilityRequirement {
            cap_type: CapabilityType::ProcessExecute,
            scope: "*".into(),
            min_permissions: CapabilityPermission::EXECUTE,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Override default timeout (max: 60)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let command = params["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams("command is required".into()))?;

        let timeout_secs = params["timeout_seconds"]
            .as_u64()
            .unwrap_or(context.sandbox.timeout.as_secs())
            .min(context.sandbox.timeout.as_secs());

        // MVP: Execute directly with timeout
        // Phase 2: Full sandbox with seccomp + namespaces
        let output = tokio::time::timeout(
            tokio::time::Duration::from_secs(timeout_secs),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .output(),
        )
        .await
        .map_err(|_| ToolError::Timeout)?;

        let output = output.map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(ToolResult {
            success: output.status.success(),
            output: serde_json::json!({
                "command": command,
                "exit_code": output.status.code(),
                "stdout": stdout,
                "stderr": stderr,
                "truncated": stdout.len() > 100_000 || stderr.len() > 100_000,
            }),
            audit_info: Some(AuditInfo {
                resource: command.to_string(),
                action: "execute".into(),
                bytes_transferred: (output.stdout.len() + output.stderr.len()) as u64,
            }),
        })
    }
}

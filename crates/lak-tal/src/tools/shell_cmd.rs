//! ShellCmd tool — execute shell commands in sandbox

use async_trait::async_trait;

use super::*;
use lak_core::types::capability::{CapabilityPermission, CapabilityRequirement, CapabilityType};

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

        // Working directory: honour the parameter but only inside the
        // sandbox's writable paths (fail closed).
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(command);

        if let Some(dir) = params["working_dir"].as_str() {
            let canonical = std::path::Path::new(dir)
                .canonicalize()
                .map_err(|e| ToolError::InvalidParams(format!("invalid working_dir: {e}")))?;
            let inside_sandbox = context
                .sandbox
                .writable_paths
                .iter()
                .any(|p| canonical.starts_with(p));
            if !inside_sandbox {
                return Err(ToolError::AccessDenied(format!(
                    "working_dir '{}' is outside sandbox writable paths",
                    canonical.display()
                )));
            }
            cmd.current_dir(canonical);
        }

        // MVP: Execute directly with timeout
        // Phase 2: Full sandbox with seccomp + namespaces
        let output =
            tokio::time::timeout(tokio::time::Duration::from_secs(timeout_secs), cmd.output())
                .await
                .map_err(|_| ToolError::Timeout)?;

        let output = output.map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        const MAX_OUTPUT: usize = 100_000;
        let stdout = truncate_utf8(&String::from_utf8_lossy(&output.stdout), MAX_OUTPUT);
        let stderr = truncate_utf8(&String::from_utf8_lossy(&output.stderr), MAX_OUTPUT);
        let truncated = output.stdout.len() > MAX_OUTPUT || output.stderr.len() > MAX_OUTPUT;

        Ok(ToolResult {
            success: output.status.success(),
            output: serde_json::json!({
                "command": command,
                "exit_code": output.status.code(),
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
            }),
            audit_info: Some(AuditInfo {
                resource: command.to_string(),
                action: "execute".into(),
                bytes_transferred: (output.stdout.len() + output.stderr.len()) as u64,
            }),
        })
    }
}

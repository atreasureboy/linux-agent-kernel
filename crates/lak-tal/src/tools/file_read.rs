//! FileRead tool — read file contents with sandbox enforcement

use async_trait::async_trait;
use std::path::Path;

use lak_core::types::capability::{CapabilityPermission, CapabilityRequirement, CapabilityType};

use super::*;

#[derive(Debug, Default)]
pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "FileRead"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns the file contents with line numbers."
    }

    fn danger_level(&self) -> DangerLevel {
        DangerLevel::Safe
    }

    fn required_capability(&self) -> CapabilityRequirement {
        CapabilityRequirement {
            cap_type: CapabilityType::FileRead,
            scope: "file:///**".into(),
            min_permissions: CapabilityPermission::READ,
        }
    }

    fn required_capability_for(&self, params: &serde_json::Value) -> CapabilityRequirement {
        let path = params["path"].as_str().unwrap_or_default();
        CapabilityRequirement {
            cap_type: CapabilityType::FileRead,
            scope: format!("file://{path}"),
            min_permissions: CapabilityPermission::READ,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 1000)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let path_str = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams("path is required".into()))?;
        let path = Path::new(path_str);

        // Resolve canonical path (symlink defense)
        let canonical = path
            .canonicalize()
            .map_err(|e| ToolError::ExecutionError(format!("Failed to resolve path: {e}")))?;

        // Path whitelist check
        let allowed = &context.sandbox.readable_paths;
        if !allowed.iter().any(|p| canonical.starts_with(p)) {
            return Err(ToolError::AccessDenied(format!(
                "Access to '{path_str}' is not allowed by sandbox policy"
            )));
        }

        let max_lines = params["max_lines"].as_u64().unwrap_or(1000) as usize;

        // File size check
        let metadata =
            std::fs::metadata(&canonical).map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        if metadata.len() > context.sandbox.max_file_read_bytes {
            return Err(ToolError::ResourceLimitExceeded(format!(
                "File too large: {} bytes (max: {})",
                metadata.len(),
                context.sandbox.max_file_read_bytes
            )));
        }

        // Read file
        let content = std::fs::read_to_string(&canonical)
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let truncated = lines.len() > max_lines;
        let displayed_lines: Vec<&str> = lines.into_iter().take(max_lines).collect();

        Ok(ToolResult {
            success: true,
            output: serde_json::json!({
                "path": path_str,
                "size_bytes": metadata.len(),
                "total_lines": total_lines,
                "lines_returned": displayed_lines.len(),
                "content": displayed_lines.join("\n"),
                "truncated": truncated,
            }),
            audit_info: Some(AuditInfo {
                resource: path_str.to_string(),
                action: "read".into(),
                bytes_transferred: metadata.len(),
            }),
        })
    }
}

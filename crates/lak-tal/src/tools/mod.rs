//! Tool trait and built-in tool implementations

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use lak_core::types::capability::CapabilityRequirement;
use lak_core::types::ids::{AgentId, TaskId};

/// 工具危险等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DangerLevel {
    /// 纯信息获取，无副作用
    Safe = 0,
    /// 信息写入，低风险
    Low = 1,
    /// 网络写入，中风险
    Medium = 2,
    /// 进程执行，高风险
    High = 3,
    /// 系统级操作，极高风险
    Critical = 4,
}

/// 工具沙箱配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub level: SandboxLevel,
    pub max_memory_mb: u64,
    pub max_cpu_seconds: u64,
    pub max_disk_mb: u64,
    pub network_policy: NetworkPolicy,
    pub writable_paths: Vec<PathBuf>,
    pub readable_paths: Vec<PathBuf>,
    pub max_file_read_bytes: u64,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxLevel {
    None,
    Light,
    Heavy,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPolicy {
    None,
    LocalhostOnly,
    Allowlist(Vec<String>),
    All,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            level: SandboxLevel::Heavy,
            max_memory_mb: 512,
            max_cpu_seconds: 30,
            max_disk_mb: 100,
            network_policy: NetworkPolicy::None,
            writable_paths: vec![PathBuf::from("/tmp/lak-sandbox")],
            readable_paths: vec![
                PathBuf::from("/tmp/lak-sandbox"),
                PathBuf::from("/workspace"),
            ],
            max_file_read_bytes: 10 * 1024 * 1024, // 10 MB
            timeout: Duration::from_secs(60),
        }
    }
}

/// 工具执行上下文
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub agent_id: AgentId,
    pub task_id: TaskId,
    pub sandbox: SandboxConfig,
}

/// 工具执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: serde_json::Value,
    pub audit_info: Option<AuditInfo>,
}

/// 工具审计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditInfo {
    pub resource: String,
    pub action: String,
    pub bytes_transferred: u64,
}

/// 工具错误
#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("Access denied: {0}")]
    AccessDenied(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Timeout")]
    Timeout,

    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}

/// Tool trait — 所有工具实现此接口
#[async_trait]
pub trait Tool: Send + Sync + std::fmt::Debug {
    /// 工具名称
    fn name(&self) -> &str;

    /// 工具描述
    fn description(&self) -> &str;

    /// 危险等级
    fn danger_level(&self) -> DangerLevel;

    /// 所需的能力（静态描述，用于展示/注册）
    fn required_capability(&self) -> CapabilityRequirement;

    /// 针对具体参数计算所需能力（用于执行时的能力检查）。
    ///
    /// 与 `required_capability` 不同，这里的 scope 是具体资源
    /// （如 `file:///etc/hosts`），因此可以用持有能力的 glob 模式
    /// 精确匹配，而不是和占位符比较。
    fn required_capability_for(&self, params: &serde_json::Value) -> CapabilityRequirement {
        let _ = params;
        self.required_capability()
    }

    /// 参数 JSON Schema
    fn parameters_schema(&self) -> serde_json::Value;

    /// 执行工具
    async fn execute(
        &self,
        params: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError>;
}

/// Truncate a string to at most `max_bytes` bytes without splitting a
/// UTF-8 code point (slicing at arbitrary byte offsets can panic).
pub(crate) fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

// ── Built-in tools ──

mod file_read;
mod http_get;
mod shell_cmd;

pub use file_read::FileReadTool;
pub use http_get::HttpGetTool;
pub use shell_cmd::ShellCmdTool;

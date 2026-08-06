//! ContextWindow — Agent Kernel 的"虚拟内存空间"
//!
//! 上下文窗口是 Agent 当前可直接访问的认知资源。
//! 类似进程的虚拟地址空间，但资源是 token 而非字节。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 上下文窗口：Agent 当前可用的认知资源配额
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    pub tokens: Vec<ContextToken>,
    pub max_tokens: usize,
    pub token_count: usize,
}

/// 上下文中的单个 token（逻辑上的内容单元）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextToken {
    pub token_id: u64,
    pub content: String,
    pub source: TokenSource,
    pub importance: f32,
    pub timestamp: DateTime<Utc>,
}

/// Token 的来源（用于安全标记和 Prompt Injection 防护）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenSource {
    /// 系统提示词（最高信任，不可被外部数据覆盖）
    SystemPrompt,
    /// 用户输入（高信任）
    UserInput,
    /// Agent 自己的思考（中信任）
    AgentThought,
    /// 工具输出（低信任——可能被攻击者控制）
    ToolOutput,
    /// 记忆检索结果（中信任）
    MemoryRetrieval,
    /// 收到的意图消息（低信任——来自其他 Agent）
    IntentReceived,
    /// 文件内容（低信任——可能含注入）
    FileContent,
}

impl TokenSource {
    /// 此来源的内容是否可信（不需要注入检测）
    pub fn is_trusted(&self) -> bool {
        matches!(self, Self::SystemPrompt | Self::UserInput)
    }

    /// 此来源是否需要注入检测
    pub fn needs_injection_check(&self) -> bool {
        matches!(
            self,
            Self::ToolOutput | Self::FileContent | Self::IntentReceived
        )
    }
}

impl ContextWindow {
    /// 创建新的上下文窗口
    pub fn new(max_tokens: usize) -> Self {
        Self {
            tokens: Vec::new(),
            max_tokens,
            token_count: 0,
        }
    }

    /// 追加内容到上下文
    pub fn append(&mut self, content: impl Into<String>, source: TokenSource) {
        let token_id = self
            .tokens
            .last()
            .map(|t| t.token_id.wrapping_add(1))
            .unwrap_or(1);
        let token = ContextToken {
            token_id,
            content: content.into(),
            source,
            importance: 1.0,
            timestamp: Utc::now(),
        };
        self.token_count += 1;
        self.tokens.push(token);
    }

    /// 上下文是否已满（需要压缩或置换）
    pub fn is_full(&self) -> bool {
        self.token_count >= self.max_tokens
    }

    /// 上下文利用率
    pub fn utilization(&self) -> f64 {
        self.token_count as f64 / self.max_tokens as f64
    }

    /// 清除上下文（Agent 重置时使用）
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.token_count = 0;
    }

    /// 保留最近 N% 的内容，其余压缩
    /// 返回被移除的 token 数
    pub fn compress(&mut self, keep_ratio: f64) -> usize {
        let keep_count = (self.tokens.len() as f64 * keep_ratio) as usize;
        let removed = self.tokens.len() - keep_count;
        self.tokens.drain(0..removed);
        self.token_count = self.tokens.len();
        removed
    }
}

/// 上下文快照（用于 Checkpoint）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub token_count: usize,
    pub max_tokens: usize,
    pub timestamp: DateTime<Utc>,
    /// 仅保存高重要性的 token 的内容摘要
    pub summary: String,
}

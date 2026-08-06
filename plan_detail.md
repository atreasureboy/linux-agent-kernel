# LAK 实施细节 — plan_detail.md

## 版本: v1.0
## 日期: 2026-08-06

---

## 前言

本文档是 [plan.md](./plan.md) 的配套实施细节。plan.md 定义了"是什么"和"为什么"，
plan_detail.md 定义"怎么做"——精确到每个文件的职责、每个函数的签名、每个测试用例。

---

## Step 0: 环境准备与项目初始化

### 0.1 前置依赖

```bash
# 系统依赖 (Ubuntu/Debian)
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    cmake \
    curl

# Rust 工具链
rustup default stable
rustup component add rustfmt clippy rust-analyzer
cargo install cargo-audit cargo-deny cargo-fuzz cargo-nextest
cargo install protoc-gen-prost  # protobuf 代码生成

# Docker (用于工具沙箱)
sudo apt install -y docker.io

# 可选: 向量数据库
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 可选: PostgreSQL + pgvector
docker run -d --name lak-postgres \
    -e POSTGRES_PASSWORD=lak \
    -p 5432:5432 \
    pgvector/pgvector:pg16
```

### 0.2 创建 Cargo Workspace

```bash
cd /project/windows_agent
cargo init --workspace

# 创建所有 crate
cargo new --lib crates/lak-core
cargo new --lib crates/lak-are
cargo new --lib crates/lak-tal
cargo new --lib crates/lak-services
cargo new --lib crates/lak-proto
cargo new crates/lakd  # binary

# 创建示例目录
mkdir -p examples tests/integration config
```

### 0.3 根 Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/lak-core",
    "crates/lak-are",
    "crates/lak-tal",
    "crates/lak-services",
    "crates/lak-proto",
    "crates/lakd",
]

[workspace.dependencies]
# 异步运行时
tokio = { version = "1.40", features = ["full"] }
async-trait = "0.1"

# 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
prost = "0.13"
tonic = "0.12"

# 工具
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
thiserror = "2"
anyhow = "1.0"

# 向量 / 数据库
qdrant-client = "1.0"
sqlx = { version = "0.8", features = ["runtime-tokio", "sqlite", "chrono"] }

# HTTP / gRPC
reqwest = { version = "0.12", features = ["json", "stream"] }
eventsource-stream = "0.2"  # SSE 流解析

# 测试
tokio-test = "0.4"
proptest = "1.0"

[workspace.lints.rust]
unsafe_code = "deny"  # 默认禁止 unsafe，核心 crate 选择性允许

[workspace.lints.clippy]
all = "warn"
pedantic = "warn"
nursery = "warn"
cargo = "warn"
```

---

## Step 1: lak-core — 核心类型系统（Day 1-4）

### 1.1 文件清单

```
crates/lak-core/
├── Cargo.toml
└── src/
    ├── lib.rs              # crate root, re-exports
    ├── types/
    │   ├── mod.rs
    │   ├── ids.rs          # AgentId, TaskId, IntentId, MemoryChunkId
    │   ├── agent.rs        # AgentSpec, AgentState, AgentConfig
    │   ├── task.rs         # CognitiveTask, TaskType, TaskState, CognitivePriority
    │   ├── intent.rs       # IntentMessage, IntentTarget, IntentType, IntentContent
    │   ├── memory.rs       # MemoryChunk, MemoryContent, MemoryMetadata, MemoryRelation
    │   ├── context.rs      # ContextWindow, ContextToken, TokenSource
    │   └── capability.rs   # Capability, CapabilityCertificate, CapabilityType, CapabilityScope
    ├── error.rs            # KernelError enum
    └── traits.rs           # AgentKernel trait
```

### 1.2 核心类型实现细节

```rust
// crates/lak-core/Cargo.toml
[package]
name = "lak-core"
version = "0.1.0"
edition = "2021"

[dependencies]
uuid.workspace = true
serde.workspace = true
chrono.workspace = true
thiserror.workspace = true
bitflags = "2.0"

// ============================================================
// crates/lak-core/src/types/ids.rs
// ============================================================

use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// 强类型 ID 宏——避免混淆不同种类的 UUID
macro_rules! define_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);
        
        impl $name {
            pub fn new() -> Self { Self(Uuid::new_v4()) }
        }
        
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

define_id!(AgentId, "智能体的唯一标识符，等同于 OS 的 PID");
define_id!(TaskId, "认知任务的唯一标识符，等同于线程的 TID");
define_id!(IntentId, "意图的唯一标识符，等同于网络包序列号");
define_id!(MemoryChunkId, "记忆片段的唯一标识符，等同于内存页号");
define_id!(CapabilityCertId, "能力证书的唯一标识符");
define_id!(CognodeId, "认知文件节点的唯一标识符");

// 特殊 ID
impl AgentId {
    pub const SUPERVISOR: Self = Self(Uuid::from_u128(1));
    pub const SYSTEM: Self = Self(Uuid::from_u128(2));
}

// ============================================================
// crates/lak-core/src/types/task.rs
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveTask {
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub task_type: TaskType,
    pub priority: CognitivePriority,
    pub state: TaskState,
    pub content: TaskContent,
    pub context_snapshot: Option<ContextSnapshot>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub dependencies: Vec<TaskId>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
    
    // 执行统计
    pub stats: TaskStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContent {
    /// 自然语言描述
    pub natural_language: String,
    /// 可选的结构化约束
    pub structured_schema: Option<serde_json::Value>,
    /// 关联的记忆 ID（上下文锚点）
    pub memory_references: Vec<MemoryChunkId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePriority {
    pub urgency: u8,           // 0-100，时间敏感度
    pub importance: u8,        // 0-100，任务重要性
    pub context_affinity: u8,  // 0-100，与当前上下文的关联度
    
    // 资源相关
    pub estimated_tokens: Option<u64>,
    pub requires_gpu: bool,
}

impl CognitivePriority {
    /// 计算优先级分数 (0.0 - 100.0)
    pub fn score(&self) -> f64 {
        (self.urgency as f64 * 0.4)
            + (self.importance as f64 * 0.4)
            + (self.context_affinity as f64 * 0.2)
    }
    
    /// 工厂方法
    pub fn low() -> Self {
        Self { urgency: 10, importance: 20, context_affinity: 10, 
               estimated_tokens: None, requires_gpu: false }
    }
    
    pub fn normal() -> Self {
        Self { urgency: 40, importance: 50, context_affinity: 50,
               estimated_tokens: None, requires_gpu: false }
    }
    
    pub fn high() -> Self {
        Self { urgency: 80, importance: 80, context_affinity: 70,
               estimated_tokens: None, requires_gpu: false }
    }
    
    pub fn critical() -> Self {
        Self { urgency: 100, importance: 100, context_affinity: 90,
               estimated_tokens: None, requires_gpu: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    Reasoning,         // LLM 推理
    ToolExecution,     // 执行工具
    MemoryRetrieval,   // 记忆检索
    IntentProcessing,  // 意图处理
    IdleReflection,    // 空闲反思
    SystemTask,        // 系统管理
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    Pending,
    Ready,
    Running,
    Blocked(String),      // 阻塞原因
    AwaitingLLM,          // 等待 LLM 响应
    AwaitingTool,         // 等待工具执行
    AwaitingIntent,       // 等待意图响应
    Suspended,
    Completed,
    Failed(TaskError),
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskStats {
    pub tokens_consumed: u64,
    pub tool_calls_made: u32,
    pub reasoning_steps: u32,
    pub memory_retrievals: u32,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub llm_calls: u32,
    pub total_wall_time_ms: u64,
}

// ============================================================
// crates/lak-core/src/types/capability.rs
// ============================================================

use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CapabilityPermission: u32 {
        const READ     = 0b0000_0001;
        const WRITE    = 0b0000_0010;
        const EXECUTE  = 0b0000_0100;
        const DELETE   = 0b0000_1000;
        const CREATE   = 0b0001_0000;
        const DELEGATE = 0b0010_0000;  // 可委派给其他 agent
        const ATTENUATE = 0b0100_0000; // 委派时可衰减
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityType {
    // 文件系统
    FileRead,
    FileWrite,
    FileDelete,
    
    // 网络
    NetworkHttp,
    NetworkWebSocket,
    
    // 进程
    ProcessExecute,
    ProcessSignal,
    
    // Agent 操作
    AgentCreate,
    AgentDestroy,
    AgentCommunicate,
    
    // 记忆操作
    MemoryRead,
    MemoryWrite,
    MemoryShare,    // 与其他 agent 共享记忆
    
    // LLM
    LLMCall,
    
    // 意图
    IntentSend,
    IntentBroadcast,
    
    // 工具
    ToolRegister,
    ToolExecute,
    
    // 系统
    SystemConfig,
    SystemMetrics,
    SystemAudit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub cap_type: CapabilityType,
    pub scope: CapabilityScope,
    pub permissions: CapabilityPermission,
    pub constraints: Vec<CapabilityConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityScope {
    /// 资源模式，支持通配符
    /// 例如: "file:///workspace/**", "http://api.github.com/*"
    pub pattern: String,
}

impl CapabilityScope {
    pub fn matches(&self, resource: &str) -> bool {
        // 使用 glob 模式匹配
        // MVP: 简单的字符串匹配，后续用 glob crate
        if self.pattern.ends_with("**") {
            let prefix = self.pattern.trim_end_matches("**");
            resource.starts_with(prefix)
        } else if self.pattern.ends_with('*') {
            let prefix = self.pattern.trim_end_matches('*');
            resource.starts_with(prefix) 
                && !resource[prefix.len()..].contains('/')
        } else {
            resource == self.pattern
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CapabilityConstraint {
    /// 每秒最大调用次数
    RateLimit { max_per_second: u32 },
    /// 总调用配额
    QuotaLimit { max_total: u64 },
    /// 时间窗口限制
    TimeWindow { 
        start_hour: u8,   // 0-23
        end_hour: u8,     // 0-23
    },
    /// 需要人工审批
    RequiresApproval,
    /// 数据大小上限
    MaxDataBytes(u64),
    /// 最大委派深度
    MaxDelegationDepth(u8),
}

/// 检查 Capability 是否满足要求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityRequirement {
    pub cap_type: CapabilityType,
    pub scope: String,
    pub min_permissions: CapabilityPermission,
}

impl Capability {
    pub fn satisfies(&self, requirement: &CapabilityRequirement) -> bool {
        self.cap_type == requirement.cap_type
            && self.scope.matches(&requirement.scope)
            && self.permissions.contains(requirement.min_permissions)
    }
    
    /// 创建衰减的能力（委派时使用）
    pub fn attenuate(
        &self, 
        new_scope: Option<CapabilityScope>,
        new_permissions: Option<CapabilityPermission>,
        additional_constraints: Vec<CapabilityConstraint>,
    ) -> Result<Self, &'static str> {
        let scope = new_scope.unwrap_or_else(|| self.scope.clone());
        let permissions = new_permissions.unwrap_or(self.permissions);
        
        // 不能扩大权限
        if !self.permissions.contains(permissions) {
            return Err("Cannot expand permissions during attenuation");
        }
        
        // 新 scope 必须在旧 scope 的子集内
        // (MVP: 简单检查，后续做严格的 glob 包含检查)
        
        let mut constraints = self.constraints.clone();
        constraints.extend(additional_constraints);
        
        Ok(Self {
            cap_type: self.cap_type.clone(),
            scope,
            permissions,
            constraints,
        })
    }
}
```

### 1.3 测试（Day 4）

```rust
// crates/lak-core/tests/capability_tests.rs

#[cfg(test)]
mod tests {
    use lak_core::types::capability::*;
    
    #[test]
    fn test_scope_matching_exact() {
        let scope = CapabilityScope { 
            pattern: "file:///workspace/readme.md".into() 
        };
        assert!(scope.matches("file:///workspace/readme.md"));
        assert!(!scope.matches("file:///workspace/secret.md"));
    }
    
    #[test]
    fn test_scope_matching_wildcard() {
        let scope = CapabilityScope { 
            pattern: "file:///workspace/*".into() 
        };
        assert!(scope.matches("file:///workspace/readme.md"));
        assert!(!scope.matches("file:///workspace/sub/readme.md"));
    }
    
    #[test]
    fn test_scope_matching_recursive() {
        let scope = CapabilityScope { 
            pattern: "file:///workspace/**".into() 
        };
        assert!(scope.matches("file:///workspace/readme.md"));
        assert!(scope.matches("file:///workspace/a/b/c/deep.txt"));
    }
    
    #[test]
    fn test_capability_satisfies() {
        let cap = Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope { pattern: "file:///workspace/**".into() },
            permissions: CapabilityPermission::READ | CapabilityPermission::DELEGATE,
            constraints: vec![],
        };
        
        let req = CapabilityRequirement {
            cap_type: CapabilityType::FileRead,
            scope: "file:///workspace/project/".into(),
            min_permissions: CapabilityPermission::READ,
        };
        assert!(cap.satisfies(&req));
    }
    
    #[test]
    fn test_capability_attenuation_cannot_expand() {
        let cap = Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope { pattern: "file:///workspace/*".into() },
            permissions: CapabilityPermission::READ,
            constraints: vec![],
        };
        
        // 尝试扩大权限——应该失败
        let result = cap.attenuate(
            None,
            Some(CapabilityPermission::READ | CapabilityPermission::WRITE),
            vec![],
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_capability_attenuation_narrow() {
        let cap = Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope { pattern: "file:///workspace/**".into() },
            permissions: CapabilityPermission::READ | CapabilityPermission::DELEGATE,
            constraints: vec![],
        };
        
        // 缩小 scope
        let attenuated = cap.attenuate(
            Some(CapabilityScope { pattern: "file:///workspace/sub/**".into() }),
            None,
            vec![CapabilityConstraint::MaxDataBytes(1024)],
        ).unwrap();
        
        assert!(attenuated.scope.matches("file:///workspace/sub/file.txt"));
        assert!(!attenuated.scope.matches("file:///workspace/other/file.txt"));
    }
    
    #[test]
    fn test_cognitive_priority_ordering() {
        let low = CognitivePriority::low();
        let high = CognitivePriority::high();
        let critical = CognitivePriority::critical();
        
        assert!(critical.score() > high.score());
        assert!(high.score() > low.score());
    }
}
```

---

## Step 2: lak-proto — Protocol Buffers（Day 5-6）

### 2.1 文件清单

```
crates/lak-proto/
├── Cargo.toml
├── build.rs               # tonic/prost 构建脚本
├── proto/
│   ├── agent.proto        # Agent 生命周期
│   ├── task.proto         # 认知任务
│   ├── intent.proto       # 意图路由
│   ├── memory.proto       # 语义记忆
│   ├── capability.proto   # 能力管理
│   ├── tool.proto         # 工具管理
│   ├── system.proto       # 系统 API
│   └── common.proto       # 共享类型
└── src/
    └── lib.rs             # include! 生成的代码, 添加辅助方法
```

### 2.2 实现

```toml
# crates/lak-proto/Cargo.toml
[package]
name = "lak-proto"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tonic-build = "0.12"

[dependencies]
tonic.workspace = true
prost.workspace = true
serde.workspace = true
```

```rust
// crates/lak-proto/build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "proto/common.proto",
                "proto/agent.proto",
                "proto/task.proto",
                "proto/intent.proto",
                "proto/memory.proto",
                "proto/capability.proto",
                "proto/tool.proto",
                "proto/system.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
```

具体 proto 定义见 plan.md v19 的完整 API 设计。

---

## Step 3: lak-tal — Tool Abstraction Layer（Day 11-16）

### 3.1 LLM Driver 实现

```rust
// crates/lak-tal/src/llm/traits.rs

use async_trait::async_trait;
use futures::stream::BoxStream;

/// LLM 后端的统一抽象
#[async_trait]
pub trait LLMDriver: Send + Sync + std::fmt::Debug {
    /// 驱动名称
    fn name(&self) -> &str;
    
    /// 支持的模型
    fn supported_models(&self) -> &[String];
    
    /// 流式生成
    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError>;
    
    /// 非流式生成
    async fn generate(
        &self,
        request: LLMRequest,
    ) -> Result<LLMResponse, LLMError> {
        let mut stream = self.generate_stream(request).await?;
        let mut tokens = Vec::new();
        let mut tool_calls = Vec::new();
        
        use futures::StreamExt;
        while let Some(event) = stream.next().await {
            match event? {
                LLMStreamEvent::Token(t) => tokens.push(t),
                LLMStreamEvent::ToolCall(tc) => tool_calls.push(tc),
                LLMStreamEvent::Done(resp) => return Ok(resp),
                LLMStreamEvent::Error(e) => return Err(e),
                _ => {}
            }
        }
        
        Ok(LLMResponse {
            content: tokens.join(""),
            tool_calls,
            tokens_used: tokens.len() as u64,
            finish_reason: "completed".into(),
        })
    }
    
    /// Token 计数
    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError>;
    
    /// 健康检查
    async fn health_check(&self) -> Result<bool, LLMError>;
    
    /// 成本估算
    fn cost_per_1k_tokens(&self, is_input: bool) -> f64;
}
```

```rust
// crates/lak-tal/src/llm/openai.rs

use async_trait::async_trait;
use reqwest::Client;
use futures::stream::BoxStream;

#[derive(Debug)]
pub struct OpenAIDriver {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    max_concurrent: usize,
}

impl OpenAIDriver {
    pub fn new(api_key: String, model: &str) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.openai.com/v1".into(),
            model: model.to_string(),
            max_concurrent: 20,
        }
    }
}

#[async_trait]
impl LLMDriver for OpenAIDriver {
    fn name(&self) -> &str { "openai" }
    
    fn supported_models(&self) -> &[String] { 
        // 静态列表
        &[] 
    }
    
    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        let messages: Vec<serde_json::Value> = request.messages.iter().map(|m| {
            serde_json::json!({
                "role": m.role.as_str(),
                "content": m.content,
            })
        }).collect();
        
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "temperature": request.temperature.unwrap_or(0.7),
        });
        
        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::to_value(tools).unwrap();
        }
        
        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;
        
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            
            if status == 429 {
                return Err(LLMError::RateLimited);
            }
            return Err(LLMError::APIError(status, body));
        }
        
        // 解析 SSE 流
        let stream = response
            .bytes_stream()
            .map(|chunk| {
                let bytes = chunk.map_err(|e| LLMError::NetworkError(e.to_string()))?;
                let text = String::from_utf8_lossy(&bytes);
                
                // 解析 SSE 行: "data: {...}"
                for line in text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            return Ok(LLMStreamEvent::Done(LLMResponse::default()));
                        }
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(choices) = parsed["choices"].as_array() {
                                if let Some(delta) = choices[0]["delta"].as_object() {
                                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                        return Ok(LLMStreamEvent::Token(content.to_string()));
                                    }
                                    if let Some(tool_calls) = delta.get("tool_calls") {
                                        return Ok(LLMStreamEvent::ToolCall(
                                            serde_json::from_value(tool_calls.clone()).unwrap()
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(LLMStreamEvent::Token(String::new()))
            });
        
        Ok(Box::pin(stream))
    }
    
    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
        // 使用 tiktoken 或近似估计
        // 粗略估计: 1 token ≈ 4 characters
        Ok(text.len() / 4)
    }
    
    async fn health_check(&self) -> Result<bool, LLMError> {
        let resp = self.client
            .get(format!("{}/models", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;
        Ok(resp.status().is_success())
    }
    
    fn cost_per_1k_tokens(&self, is_input: bool) -> f64 {
        match (self.model.as_str(), is_input) {
            ("gpt-4o", true) => 0.0025,
            ("gpt-4o", false) => 0.01,
            ("gpt-4o-mini", true) => 0.00015,
            ("gpt-4o-mini", false) => 0.0006,
            _ => 0.0,
        }
    }
}
```

### 3.2 Tool Trait 实现

```rust
// crates/lak-tal/src/tools/mod.rs

#[async_trait]
pub trait Tool: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn danger_level(&self) -> DangerLevel;
    fn required_capability(&self) -> CapabilityRequirement;
    
    async fn execute(
        &self,
        params: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError>;
}

#[derive(Debug, Clone)]
pub struct ToolContext {
    pub agent_id: AgentId,
    pub task_id: TaskId,
    pub sandbox: SandboxConfig,
    pub capabilities: Vec<Capability>,
    pub working_dir: PathBuf,
    pub timeout: Duration,
}

// crates/lak-tal/src/tools/file_read.rs

#[derive(Debug)]
pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str { "FileRead" }
    fn description(&self) -> &str { "Read the contents of a file" }
    fn danger_level(&self) -> DangerLevel { DangerLevel::Safe }
    
    fn required_capability(&self) -> CapabilityRequirement {
        CapabilityRequirement {
            cap_type: CapabilityType::FileRead,
            scope: "{path}".into(),  // 路径在运行时检查
            min_permissions: CapabilityPermission::READ,
        }
    }
    
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default 1000)"
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
        let path_str = params["path"].as_str()
            .ok_or(ToolError::InvalidParams("path is required".into()))?;
        let path = Path::new(path_str);
        
        // 安全: 路径必须在允许的范围内
        let canonical = path.canonicalize()
            .map_err(|e| ToolError::ExecutionError(format!("Failed to resolve path: {e}")))?;
        
        // 路径白名单检查
        let allowed = &context.sandbox.readable_paths;
        if !allowed.iter().any(|p| canonical.starts_with(p)) {
            return Err(ToolError::AccessDenied(format!(
                "Access to {path_str} is not allowed"
            )));
        }
        
        let max_lines = params["max_lines"].as_u64().unwrap_or(1000) as usize;
        
        // 读取文件（带大小限制）
        let metadata = tokio::fs::metadata(&canonical).await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
        
        if metadata.len() > context.sandbox.max_file_read_bytes {
            return Err(ToolError::ExecutionError(
                format!("File too large: {} bytes", metadata.len())
            ));
        }
        
        let content = tokio::fs::read_to_string(&canonical).await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
        
        // 截断到最大行数
        let lines: Vec<&str> = content.lines().take(max_lines).collect();
        let truncated = lines.join("\n");
        
        Ok(ToolResult {
            success: true,
            output: serde_json::json!({
                "path": path_str,
                "size_bytes": metadata.len(),
                "lines_read": lines.len(),
                "content": truncated,
                "truncated": lines.len() < content.lines().count(),
            }),
            audit_info: Some(AuditInfo {
                resource: path_str.to_string(),
                action: "read".into(),
                bytes_transferred: metadata.len(),
            }),
        })
    }
}
```

---

## Step 4: lak-services — Agent Services（Day 17-20）

### 4.1 文件清单

```
crates/lak-services/
└── src/
    ├── lib.rs
    ├── reasoning/
    │   ├── mod.rs
    │   ├── service.rs        # ReasoningService
    │   └── model_router.rs   # ModelRouter
    ├── memory/
    │   ├── mod.rs
    │   ├── service.rs        # MemoryService
    │   └── store.rs          # MemoryStore trait + implementations
    └── tool_registry.rs      # ToolRegistry
```

### 4.2 ReasoningService

```rust
// crates/lak-services/src/reasoning/service.rs

pub struct ReasoningService {
    drivers: HashMap<String, Arc<dyn LLMDriver>>,
    model_router: ModelRouter,
    cost_tracker: Arc<CostTracker>,
}

impl ReasoningService {
    pub async fn execute_task(
        &self,
        task: &CognitiveTask,
        agent_process: &AgentProcess,
        context: &ContextWindow,
    ) -> Result<TaskResult, ReasoningError> {
        // 1. 选择模型
        let model = self.model_router.select(task)?;
        
        // 2. 构建 LLM 请求
        let request = self.build_request(task, agent_process, context)?;
        
        // 3. 发送给 LLM
        let response = model.generate(request).await?;
        
        // 4. 处理工具调用
        let tool_results = self.handle_tool_calls(&response).await?;
        
        // 5. 如果 LLM 请求了工具，可能需要多轮
        // (简化版: 单轮，后续扩展为 agentic loop)
        
        // 6. 成本记录
        self.cost_tracker.record(task.agent_id, response.tokens_used, model.name()).await;
        
        Ok(TaskResult {
            content: response.content,
            tool_results,
            tokens_used: response.tokens_used,
            model_used: model.name().to_string(),
            estimated_cost_usd: self.cost_tracker.estimate_cost(response.tokens_used, model.name()),
        })
    }
}
```

---

## Step 5: lak-are — Agent Runtime Environment（Day 21-24）

```rust
// crates/lak-are/src/process.rs

pub struct AgentProcess {
    pub agent_id: AgentId,
    pub spec: AgentSpec,
    pub state: AgentState,
    pub context: ContextWindow,
    pub capabilities: CapabilityCertificate,
    pub active_tasks: HashMap<TaskId, CognitiveTask>,
    pub memory_layer: WorkingMemory,
    pub stats: AgentStats,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_active_at: chrono::DateTime<chrono::Utc>,
}

impl AgentProcess {
    pub fn new(spec: AgentSpec, capabilities: CapabilityCertificate) -> Self {
        Self {
            agent_id: AgentId::new(),
            spec,
            state: AgentState::Created,
            context: ContextWindow::new(32768),
            capabilities,
            active_tasks: HashMap::new(),
            memory_layer: WorkingMemory::new(),
            stats: AgentStats::default(),
            created_at: chrono::Utc::now(),
            last_active_at: chrono::Utc::now(),
        }
    }
    
    /// 执行一个认知周期
    pub async fn execute_cycle(
        &mut self,
        scheduler: &CognitiveScheduler,
        memory_service: &MemoryService,
        reasoning_service: &ReasoningService,
        tool_registry: &ToolRegistry,
    ) -> Result<CycleResult, KernelError> {
        // 1. 从调度器获取下一个任务
        let task = scheduler.dequeue_task(self.agent_id).await?;
        
        // 2. 更新状态
        self.state = AgentState::Running;
        self.last_active_at = chrono::Utc::now();
        
        // 3. 检索相关记忆
        let memories = memory_service.query(
            self.agent_id,
            &task.content.natural_language,
            10,
        ).await?;
        
        // 4. 将记忆加载到上下文
        for mem in &memories {
            self.context.append(
                format!("[Memory] {}", mem.content.raw_text),
                TokenSource::MemoryRetrieval,
            );
        }
        
        // 5. 执行推理
        let result = reasoning_service.execute_task(
            &task,
            self,
            &self.context,
        ).await?;
        
        // 6. 将结果写入上下文
        self.context.append(
            result.content.clone(),
            TokenSource::AgentThought,
        );
        
        // 7. 可能存储新的记忆
        if result.should_remember() {
            memory_service.store(
                self.agent_id,
                MemoryContent::from_reasoning(&result),
            ).await?;
        }
        
        // 8. 更新统计
        self.stats.total_tasks_completed += 1;
        self.stats.total_tokens_consumed += result.tokens_used;
        
        // 9. 如果无更多任务，进入 IDLE
        if scheduler.pending_task_count(self.agent_id) == 0 {
            self.state = AgentState::Idle;
        }
        
        Ok(CycleResult { task_id: task.task_id, result, memories_retrieved: memories.len() })
    }
}
```

---

## Step 6: lakd — Daemon 主进程（Day 27-30）

```rust
// crates/lakd/src/main.rs

use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;

mod server;
mod config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化 tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("lak=debug".parse()?)
                .add_directive("tonic=info".parse()?)
        )
        .json()
        .init();
    
    tracing::info!("[LAK] Linux Agent Kernel Daemon starting...");
    tracing::info!("[LAK] Version: {}", env!("CARGO_PKG_VERSION"));
    
    // 加载配置
    let config = config::Config::load("config/lakd.yaml")?;
    tracing::info!("[LAK] Configuration loaded");
    
    // 初始化核心组件
    let (kernel, tal, services) = initialize_components(&config).await?;
    tracing::info!("[LAK] Core components initialized");
    
    // 启动内置服务
    let supervisor_id = services.supervisor.start().await?;
    tracing::info!("[LAK] SupervisorAgent started: {}", supervisor_id);
    
    // 启动 gRPC 服务器
    let grpc_server = server::AgentKernelServer::new(
        kernel.clone(),
        services.clone(),
    );
    
    let addr = config.server.listen_addr.parse()?;
    tracing::info!("[LAK] gRPC server listening on {}", addr);
    
    // 优雅关闭
    Server::builder()
        .add_service(
            lak_proto::agent_kernel_server::AgentKernelServer::new(grpc_server)
        )
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;
    
    tracing::info!("[LAK] Shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for CTRL+C");
    tracing::info!("[LAK] Shutdown signal received");
}

async fn initialize_components(
    config: &config::Config,
) -> Result<(
    Arc<lak_core::Kernel>,
    Arc<lak_tal::ToolAbstractionLayer>,
    Arc<lak_services::Services>,
), Box<dyn std::error::Error>> {
    // 1. TAL (LLM drivers + tools)
    let tal = lak_tal::ToolAbstractionLayer::new(&config.tal).await?;
    
    // 2. Core Kernel
    let kernel = lak_core::Kernel::new(config.core.clone()).await?;
    
    // 3. Services
    let services = lak_services::Services::new(
        kernel.clone(),
        tal.clone(),
        &config.services,
    ).await?;
    
    Ok((kernel, tal, services))
}
```

---

## Step 7: 集成测试与端到端验证（Day 31-35）

```rust
// tests/integration/end_to_end.rs

#[tokio::test]
async fn test_create_agent_and_complete_task() {
    // 1. 启动 LAKd 测试实例
    let lakd = LakdTestInstance::start().await.unwrap();
    
    // 2. 创建 gRPC 客户端
    let mut client = lakd.client().await.unwrap();
    
    // 3. 创建 Agent
    let agent = client.create_agent(tonic::Request::new(
        CreateAgentRequest {
            name: "TestAgent".into(),
            system_prompt: "You are a helpful test agent. Respond concisely.".into(),
            model: "test-mock".into(),  // 使用 mock LLM
            max_context_tokens: 4096,
            initial_capabilities: vec![
                lak_proto::Capability {
                    cap_type: lak_proto::CapabilityType::FileRead as i32,
                    scope: Some(lak_proto::CapabilityScope {
                        pattern: "file:///tmp/lak-test/**".into(),
                    }),
                    permissions: 1, // READ
                    constraints: vec![],
                }
            ],
            memory_quota_bytes: 1024 * 1024,
            ..Default::default()
        }
    )).await.unwrap().into_inner();
    
    assert!(!agent.agent_id.is_empty());
    assert_eq!(agent.state, lak_proto::AgentState::Created as i32);
    
    // 4. 提交任务
    let task = client.submit_task(tonic::Request::new(
        SubmitTaskRequest {
            agent_id: agent.agent_id.clone(),
            content: "Say 'Hello, LAK!'".into(),
            priority: Some(lak_proto::CognitivePriority {
                urgency: 50,
                importance: 50,
            }),
            ..Default::default()
        }
    )).await.unwrap().into_inner();
    
    assert!(!task.task_id.is_empty());
    
    // 5. 等待任务完成
    let mut stream = client.watch_task(tonic::Request::new(
        WatchTaskRequest { task_id: task.task_id.clone() }
    )).await.unwrap().into_inner();
    
    let mut completed = false;
    while let Some(event) = stream.message().await.unwrap() {
        match lak_proto::TaskStatus::try_from(event.status).unwrap() {
            lak_proto::TaskStatus::Completed => {
                assert!(event.result.unwrap().content.contains("Hello, LAK!"));
                completed = true;
                break;
            }
            lak_proto::TaskStatus::Failed => {
                panic!("Task failed: {:?}", event.error);
            }
            _ => {}
        }
    }
    assert!(completed, "Task did not complete");
    
    // 6. 验证 Agent 统计
    let agent = client.get_agent(tonic::Request::new(
        GetAgentRequest { agent_id: agent.agent_id.clone() }
    )).await.unwrap().into_inner();
    
    assert_eq!(agent.stats.unwrap().total_tasks_completed, 1);
    
    // 7. 清理
    client.destroy_agent(tonic::Request::new(
        DestroyAgentRequest { agent_id: agent.agent_id.clone() }
    )).await.unwrap();
    
    lakd.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_capability_enforcement() {
    let lakd = LakdTestInstance::start().await.unwrap();
    let mut client = lakd.client().await.unwrap();
    
    // 创建没有 FileRead 能力的 Agent
    let agent = client.create_agent(tonic::Request::new(
        CreateAgentRequest {
            name: "NoFileAccess".into(),
            system_prompt: "You cannot read files.".into(),
            model: "test-mock".into(),
            max_context_tokens: 4096,
            initial_capabilities: vec![], // 空能力集
            ..Default::default()
        }
    )).await.unwrap().into_inner();
    
    // 尝试执行文件读取工具——应该被拒绝
    let result = client.execute_tool(tonic::Request::new(
        ExecuteToolRequest {
            agent_id: agent.agent_id.clone(),
            tool_name: "FileRead".into(),
            params: serde_json::json!({"path": "/tmp/test.txt"}).to_string(),
        }
    )).await;
    
    // 应该返回 PERMISSION_DENIED
    assert!(result.is_err() || 
            result.unwrap().into_inner().error_code == "CAPABILITY_DENIED");
    
    lakd.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_intent_routing() {
    let lakd = LakdTestInstance::start().await.unwrap();
    let mut client = lakd.client().await.unwrap();
    
    // 创建两个 Agent
    let agent_a = client.create_agent(/* ... */).await.unwrap().into_inner();
    let agent_b = client.create_agent(/* ... */).await.unwrap().into_inner();
    
    // Agent A 发送 Intent 给 Agent B
    let intent = client.send_intent(tonic::Request::new(
        SendIntentRequest {
            source_agent_id: agent_a.agent_id.clone(),
            target: Some(lak_proto::IntentTarget {
                target: Some(lak_proto::intent_target::Target::AgentId(
                    agent_b.agent_id.clone()
                )),
            }),
            intent_type: lak_proto::IntentType::Query as i32,
            natural_language: "What is your status?".into(),
            ..Default::default()
        }
    )).await.unwrap().into_inner();
    
    assert!(!intent.intent_id.is_empty());
    
    // Agent B 应该能接收到 Intent
    let mut stream = client.await_intent(tonic::Request::new(
        AwaitIntentRequest { 
            agent_id: agent_b.agent_id.clone(),
            timeout_seconds: 5,
        }
    )).await.unwrap().into_inner();
    
    let received = stream.message().await.unwrap().unwrap();
    assert_eq!(received.source_agent_id, agent_a.agent_id);
    assert_eq!(received.natural_language, "What is your status?");
    
    lakd.shutdown().await.unwrap();
}
```

---

## Step 8: CI/CD + Docker（Day 33-35）

### 8.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      
      - name: Install protoc
        run: sudo apt-get install -y protobuf-compiler
      
      - name: Cache
        uses: Swatinem/rust-cache@v2
      
      - name: Format
        run: cargo fmt --all -- --check
      
      - name: Clippy
        run: cargo clippy --all-targets -- -D warnings
      
      - name: Audit
        run: cargo audit
      
      - name: Check unused deps
        run: cargo deny check

  test:
    runs-on: ubuntu-latest
    needs: check
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Install protoc
        run: sudo apt-get install -y protobuf-compiler
      
      - name: Cache
        uses: Swatinem/rust-cache@v2
      
      - name: Test
        run: cargo nextest run --all-features
      
      - name: Doc tests
        run: cargo test --doc
      
      - name: Security audit
        run: cargo audit

  docker:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t lakd:latest .
      
      - name: Push to registry
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker tag lakd:latest ghcr.io/${{ github.repository }}/lakd:latest
          docker push ghcr.io/${{ github.repository }}/lakd:latest
```

### 8.2 Dockerfile

```dockerfile
# Dockerfile
FROM rust:1.80-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release -p lakd

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/lakd /usr/local/bin/lakd
COPY config/lakd.yaml /etc/lak/lakd.yaml

# 创建非 root 用户
RUN useradd -m -u 1000 lak
USER lak

EXPOSE 9191
ENTRYPOINT ["lakd", "--config", "/etc/lak/lakd.yaml"]
```

---

## 项目文件总清单

```
windows_agent/
├── Cargo.toml                    # Workspace root
├── Cargo.lock
├── .gitignore
├── Dockerfile
├── plan.md                       # 20 轮设计迭代
├── plan_detail.md                # ← 本文件
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
│   └── lakd.yaml                 # 默认配置
├── crates/
│   ├── lak-core/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── error.rs
│   │   │   ├── traits.rs
│   │   │   └── types/
│   │   │       ├── mod.rs
│   │   │       ├── ids.rs
│   │   │       ├── agent.rs
│   │   │       ├── task.rs
│   │   │       ├── intent.rs
│   │   │       ├── memory.rs
│   │   │       ├── context.rs
│   │   │       └── capability.rs
│   │   └── tests/
│   │       ├── capability_tests.rs
│   │       ├── task_tests.rs
│   │       └── integration/
│   ├── lak-proto/
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   ├── proto/
│   │   │   ├── common.proto
│   │   │   ├── agent.proto
│   │   │   ├── task.proto
│   │   │   ├── intent.proto
│   │   │   ├── memory.proto
│   │   │   ├── capability.proto
│   │   │   ├── tool.proto
│   │   │   └── system.proto
│   │   └── src/
│   │       └── lib.rs
│   ├── lak-tal/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── llm/
│   │       │   ├── mod.rs
│   │       │   ├── traits.rs
│   │       │   ├── openai.rs
│   │       │   ├── anthropic.rs
│   │       │   └── ollama.rs
│   │       └── tools/
│   │           ├── mod.rs
│   │           ├── file_read.rs
│   │           ├── file_write.rs
│   │           ├── shell_cmd.rs
│   │           ├── http_get.rs
│   │           └── sandbox.rs
│   ├── lak-services/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── reasoning/
│   │       │   ├── mod.rs
│   │       │   ├── service.rs
│   │       │   └── model_router.rs
│   │       ├── memory/
│   │       │   ├── mod.rs
│   │       │   ├── service.rs
│   │       │   └── store.rs
│   │       └── tool_registry.rs
│   ├── lak-are/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── process.rs
│   │       ├── context.rs
│   │       └── intent_parser.rs
│   └── lakd/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           ├── server.rs
│           └── config.rs
├── examples/
│   ├── simple_agent.rs
│   └── multi_agent.rs
└── tests/
    └── integration/
        ├── end_to_end.rs
        ├── capability_test.rs
        └── intent_test.rs
```

---

## 第一周实施检查清单

```
Day 1:
  □ 安装依赖 (Rust, protoc, Docker, Qdrant)
  □ cargo init --workspace
  □ 创建所有 crate 目录
  □ 编写根 Cargo.toml

Day 2:
  □ 实现 lak-core types/ids.rs (所有 ID 类型)
  □ 实现 lak-core types/task.rs (CognitiveTask)
  □ 实现 lak-core types/capability.rs (Capability 系统)
  □ 编写单元测试

Day 3:
  □ 实现 lak-core types/intent.rs (IntentMessage)
  □ 实现 lak-core types/memory.rs (MemoryChunk)
  □ 实现 lak-core error.rs (KernelError)
  □ 实现 lak-core traits.rs (AgentKernel trait)

Day 4:
  □ 完善所有类型测试
  □ 编写 Capability 系统的 proptest
  □ cargo test — 全部通过

Day 5:
  □ 安装 protoc, 配置 tonic-build
  □ 编写所有 .proto 文件
  □ 验证 protobuf 编译通过

Day 6:
  □ 编写 proto 辅助方法
  □ proto ↔ core type 转换
  □ 生成 proto 文档

Day 7:
  □ 回顾第一周进度
  □ 修复类型系统问题
  □ 准备第二周 LLM 集成
```

---

*plan_detail.md — 实现蓝图的终点，工程的起点。*

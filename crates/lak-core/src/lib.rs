//! Linux Agent Kernel (LAK) — Core
//!
//! 智能体内核的核心类型系统和接口定义。
//! 这是整个 LAK 项目的基础 crate。

pub mod error;
pub mod token_budget;
pub mod traits;
pub mod types;

pub use error::KernelError;
pub use traits::{AgentKernel, SystemStatus};

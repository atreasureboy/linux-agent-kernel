//! Capability 系统 — Agent Kernel 的"权限模型"
//!
//! 替代传统 ACL：每个 Agent 拥有 CapabilityCertificate，
//! 能力可以被委派、衰减、有时效。
//! 没有能力 = 操作被拒绝，无论 Agent "是谁"。

use bitflags::bitflags;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::ids::{AgentId, CapabilityCertId};

bitflags! {
    /// 能力权限标志
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct CapabilityPermission: u32 {
        const READ       = 0b0000_0001;
        const WRITE      = 0b0000_0010;
        const EXECUTE    = 0b0000_0100;
        const DELETE     = 0b0000_1000;
        const CREATE     = 0b0001_0000;
        /// 可以将此能力委派给其他 Agent
        const DELEGATE   = 0b0010_0000;
        /// 委派时可以衰减（缩小范围/增加约束）
        const ATTENUATE  = 0b0100_0000;
        /// 管理权限（修改此能力本身）
        const ADMIN      = 0b1000_0000;
    }
}

/// 能力类型——Agent 可以做什么
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityType {
    // 文件系统
    FileRead,
    FileWrite,
    FileDelete,
    FileWatch,

    // 网络
    NetworkHttp,
    NetworkWebSocket,
    NetworkTcp,

    // 进程
    ProcessExecute,
    ProcessSignal,

    // Agent 管理
    AgentCreate,
    AgentDestroy,
    AgentCommunicate,

    // 记忆
    MemoryRead,
    MemoryWrite,
    MemoryShare, // 与其他 Agent 共享记忆

    // LLM
    LLMCall,
    LLMFineTune, // 微调模型（高危）

    // 意图
    IntentSend,
    IntentBroadcast,

    // 工具
    ToolRegister,
    ToolExecute,

    // 认知文件系统
    CognodeRead,
    CognodeWrite,

    // 系统
    SystemConfig,
    SystemMetrics,
    SystemAudit,
    SystemShutdown,
}

/// 单个能力：类型 + 作用域 + 权限 + 约束
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub cap_type: CapabilityType,
    pub scope: CapabilityScope,
    pub permissions: CapabilityPermission,
    pub constraints: Vec<CapabilityConstraint>,
}

/// 能力的作用域（用 glob 模式定义资源范围）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityScope {
    /// 资源模式，支持通配符
    /// 例: "file:///workspace/**", "http://api.github.com/*"
    pub pattern: String,
}

impl CapabilityScope {
    /// 检查给定资源是否在此作用域内
    pub fn matches(&self, resource: &str) -> bool {
        // 使用 glob 模式匹配
        // 对于 **: 匹配任意深度
        // 对于 *: 匹配单层
        if self.pattern.ends_with("**") {
            let prefix = self.pattern.trim_end_matches("**");
            return resource.starts_with(prefix);
        }
        if self.pattern.ends_with('*') {
            let prefix = self.pattern.trim_end_matches('*');
            if !resource.starts_with(prefix) {
                return false;
            }
            let remaining = &resource[prefix.len()..];
            return !remaining.contains('/');
        }
        resource == self.pattern
    }
}

/// 能力的约束条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CapabilityConstraint {
    /// 每秒最大调用次数
    RateLimit { max_per_second: u32 },
    /// 总调用配额（用完即止）
    QuotaLimit { max_total: u64 },
    /// 时间窗口限制
    TimeWindow {
        /// 允许的起始小时 (0-23 UTC)
        start_hour: u8,
        /// 允许的结束小时 (0-23 UTC)
        end_hour: u8,
    },
    /// 需要人工审批
    RequiresApproval,
    /// 最大数据量（字节）
    MaxDataBytes(u64),
    /// 最大委派深度
    MaxDelegationDepth(u8),
    /// 仅允许本地访问
    LocalOnly,
}

/// 能力证书——Agent 持有的"通行证"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityCertificate {
    pub cert_id: CapabilityCertId,
    /// 证书持有者
    pub agent_id: AgentId,
    /// 证书签发者
    pub issued_by: AgentId,
    /// 签发时间
    pub issued_at: DateTime<Utc>,
    /// 过期时间（None = 永不过期）
    pub expires_at: Option<DateTime<Utc>>,
    /// 持有的能力列表
    pub capabilities: Vec<Capability>,
    /// 父证书 ID（如果是委派来的）
    pub parent_cert_id: Option<CapabilityCertId>,
}

/// 能力需求——操作需要的权限
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityRequirement {
    pub cap_type: CapabilityType,
    pub scope: String,
    pub min_permissions: CapabilityPermission,
}

// ── Capability 的方法 ──

impl Capability {
    /// 检查此能力是否满足给定的需求
    pub fn satisfies(&self, requirement: &CapabilityRequirement) -> bool {
        self.cap_type == requirement.cap_type
            && self.scope.matches(&requirement.scope)
            && self.permissions.contains(requirement.min_permissions)
    }

    /// 创建衰减版的能力（委派时使用）
    ///
    /// 衰减规则：
    /// - 新的 scope 必须是原 scope 的子集
    /// - 新的 permissions 必须是原 permissions 的子集
    /// - 可以添加额外的约束
    pub fn attenuate(
        &self,
        new_scope: Option<CapabilityScope>,
        new_permissions: Option<CapabilityPermission>,
        additional_constraints: Vec<CapabilityConstraint>,
    ) -> Result<Self, AttenuationError> {
        let scope = new_scope.unwrap_or_else(|| self.scope.clone());
        let permissions = new_permissions.unwrap_or(self.permissions);

        // 不能扩大权限
        if !self.permissions.contains(permissions) {
            return Err(AttenuationError::PermissionExpansion);
        }

        // 不能扩大范围（简单检查：新 pattern 比旧 pattern 更长/更具体）
        if scope.pattern.len() > self.scope.pattern.len()
            && !scope.pattern.starts_with(
                self.scope
                    .pattern
                    .trim_end_matches("**")
                    .trim_end_matches('*'),
            )
        {
            return Err(AttenuationError::ScopeExpansion);
        }

        let mut constraints = self.constraints.clone();
        constraints.extend(additional_constraints);

        Ok(Self {
            cap_type: self.cap_type.clone(),
            scope,
            permissions,
            constraints,
        })
    }

    /// 是否可以委派
    pub fn is_delegatable(&self) -> bool {
        self.permissions.contains(CapabilityPermission::DELEGATE)
    }

    /// 检查时间窗口约束
    pub fn is_within_time_window(&self, now_hour: u8) -> bool {
        for constraint in &self.constraints {
            if let CapabilityConstraint::TimeWindow {
                start_hour,
                end_hour,
            } = constraint
            {
                if *start_hour <= *end_hour {
                    if now_hour < *start_hour || now_hour > *end_hour {
                        return false;
                    }
                } else {
                    // 跨越午夜的窗口 (如 22:00 - 06:00)
                    if now_hour < *start_hour && now_hour > *end_hour {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl CapabilityCertificate {
    /// 检查证书是否有效
    pub fn is_valid(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            if Utc::now() > expires_at {
                return false;
            }
        }
        true
    }

    /// 检查是否具有某个能力
    pub fn has_capability(&self, requirement: &CapabilityRequirement) -> bool {
        if !self.is_valid() {
            return false;
        }
        self.capabilities.iter().any(|c| c.satisfies(requirement))
    }

    /// 查找满足需求的能力（用于委派时找到源能力）
    pub fn find_capability(&self, requirement: &CapabilityRequirement) -> Option<&Capability> {
        self.capabilities
            .iter()
            .find(|c| c.satisfies(requirement) && c.is_delegatable())
    }
}

/// 能力衰减时的错误
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AttenuationError {
    #[error("Cannot expand permissions during attenuation")]
    PermissionExpansion,
    #[error("Cannot expand scope during attenuation")]
    ScopeExpansion,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_exact_match() {
        let scope = CapabilityScope {
            pattern: "file:///workspace/readme.md".into(),
        };
        assert!(scope.matches("file:///workspace/readme.md"));
        assert!(!scope.matches("file:///workspace/other.md"));
    }

    #[test]
    fn test_scope_wildcard_single_level() {
        let scope = CapabilityScope {
            pattern: "file:///workspace/*".into(),
        };
        assert!(scope.matches("file:///workspace/readme.md"));
        assert!(!scope.matches("file:///workspace/sub/readme.md"));
    }

    #[test]
    fn test_scope_wildcard_recursive() {
        let scope = CapabilityScope {
            pattern: "file:///workspace/**".into(),
        };
        assert!(scope.matches("file:///workspace/readme.md"));
        assert!(scope.matches("file:///workspace/a/b/c/deep.txt"));
        assert!(!scope.matches("file:///other/file.txt"));
    }

    #[test]
    fn test_capability_satisfies() {
        let cap = Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope {
                pattern: "file:///workspace/**".into(),
            },
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
    fn test_capability_does_not_satisfy_wrong_type() {
        let cap = Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope {
                pattern: "file:///**".into(),
            },
            permissions: CapabilityPermission::READ,
            constraints: vec![],
        };

        let req = CapabilityRequirement {
            cap_type: CapabilityType::FileWrite,
            scope: "file:///workspace/".into(),
            min_permissions: CapabilityPermission::WRITE,
        };
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn test_attenuation_cannot_expand_permissions() {
        let cap = Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope {
                pattern: "file:///*".into(),
            },
            permissions: CapabilityPermission::READ,
            constraints: vec![],
        };
        let result = cap.attenuate(
            None,
            Some(CapabilityPermission::READ | CapabilityPermission::WRITE),
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_attenuation_narrow_scope() {
        let cap = Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope {
                pattern: "file:///workspace/**".into(),
            },
            permissions: CapabilityPermission::READ
                | CapabilityPermission::DELEGATE
                | CapabilityPermission::ATTENUATE,
            constraints: vec![],
        };

        let attenuated = cap
            .attenuate(
                Some(CapabilityScope {
                    pattern: "file:///workspace/project/**".into(),
                }),
                Some(CapabilityPermission::READ),
                vec![CapabilityConstraint::MaxDataBytes(1024)],
            )
            .unwrap();

        assert!(attenuated
            .scope
            .matches("file:///workspace/project/file.txt"));
        assert!(!attenuated.scope.matches("file:///workspace/other/file.txt"));
        assert_eq!(attenuated.constraints.len(), 1);
    }

    #[test]
    fn test_certificate_expiry() {
        let cert = CapabilityCertificate {
            cert_id: CapabilityCertId::new(),
            agent_id: AgentId::new(),
            issued_by: AgentId::SUPERVISOR,
            issued_at: Utc::now(),
            expires_at: Some(Utc::now() - chrono::Duration::hours(1)),
            capabilities: vec![],
            parent_cert_id: None,
        };
        assert!(!cert.is_valid());
    }

    #[test]
    fn test_time_window_constraint() {
        let cap = Capability {
            cap_type: CapabilityType::NetworkHttp,
            scope: CapabilityScope {
                pattern: "http://*".into(),
            },
            permissions: CapabilityPermission::READ,
            constraints: vec![CapabilityConstraint::TimeWindow {
                start_hour: 9,
                end_hour: 17,
            }],
        };

        assert!(cap.is_within_time_window(10)); // 上午 10 点
        assert!(!cap.is_within_time_window(3)); // 凌晨 3 点
        assert!(!cap.is_within_time_window(20)); // 晚上 8 点
    }

    #[test]
    fn test_certificate_has_capability() {
        let cert = CapabilityCertificate {
            cert_id: CapabilityCertId::new(),
            agent_id: AgentId::new(),
            issued_by: AgentId::SUPERVISOR,
            issued_at: Utc::now(),
            expires_at: None,
            capabilities: vec![Capability {
                cap_type: CapabilityType::FileRead,
                scope: CapabilityScope {
                    pattern: "file:///workspace/**".into(),
                },
                permissions: CapabilityPermission::READ,
                constraints: vec![],
            }],
            parent_cert_id: None,
        };

        assert!(cert.has_capability(&CapabilityRequirement {
            cap_type: CapabilityType::FileRead,
            scope: "file:///workspace/data.csv".into(),
            min_permissions: CapabilityPermission::READ,
        }));

        assert!(!cert.has_capability(&CapabilityRequirement {
            cap_type: CapabilityType::FileWrite,
            scope: "file:///workspace/data.csv".into(),
            min_permissions: CapabilityPermission::WRITE,
        }));
    }
}

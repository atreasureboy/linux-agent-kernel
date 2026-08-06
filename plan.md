# Linux Agent Kernel (LAK) — 智能体内核

## 版本: Plan v20 — 最终综合 + 完整决策汇总
## 日期: 2026-08-06

---

## 一、20 轮迭代全景

```
╔═══════════════════════════════════════════════════════════════╗
║          LAK 设计迭代全景 — 从科幻到可执行的蓝图                 ║
╠═════╦═════════════════╦══════════════════════════════════════╣
║  v1 ║ 概念框架         ║ 四层架构、传统 vs Agent 内核对比      ║
║  v2 ║ 调度 + 意图      ║ 三层调度器、COI、ThinkingQuantum    ║
║  v3 ║ 记忆 + 性能      ║ 四层记忆、S-Clock、延迟预算          ║
║  v4 ║ 安全 + 协作      ║ Capability、WFG、OS 共存三阶段      ║
║  v5 ║ MVP 方案         ║ Rust 选型、7 crates、gRPC API      ║
║  v6 ║ NT 映射          ║ Handle→Capability、IRP→IRQP 等15组  ║
║  v7 ║ 推理微架构       ║ 认知流水线、投机推理、多推理链      ║
║  v8 ║ 故障韧性         ║ 认知故障、Cognitive Journal、自愈    ║
║  v9 ║ 认知文件系统     ║ Cognode、SemanticSearch             ║
║ v10 ║ 实现路线图       ║ MVPP、35天计划、第一行代码           ║
╠═════╬═════════════════╬══════════════════════════════════════╣
║ v11 ║ 平台决策         ║ WAK→LAK、Linux-first、可行性矩阵    ║
║ v12 ║ 威胁建模         ║ STRIDE+A、6大攻击面、Prompt注入      ║
║ v13 ║ 形式化验证       ║ TLA+规范、seL4经验、unsafe审计      ║
║ v14 ║ LLM 现实         ║ 延迟/成本数据、混合模型、弹性调用    ║
║ v15 ║ 工具安全         ║ 5级危险分类、Heavy沙箱、审批流程    ║
║ v16 ║ 内核集成         ║ CognitiveSchedClass、eBPF、syscall  ║
║ v17 ║ 分布式集群       ║ 集群架构、Agent迁移、联邦记忆        ║
║ v18 ║ 运维监控         ║ 三层可观测、告警、备份恢复、DR       ║
║ v19 ║ API + SDK        ║ gRPC API、配置规范、Python+Rust SDK ║
║ v20 ║ 最终综合         ║ 决策汇总、实施计划、plan_detail      ║
╚═════╩═════════════════╩══════════════════════════════════════╝
```

---

## 二、20 轮迭代中的关键决策汇总

### 决策 1: 平台 = Linux-first（v11）
**理由：** Windows NT 内核源码闭源不可得，Linux 内核完全开源且 Rust-for-Linux 已合并主线
**影响：** 项目更名 LAK，Phase 1 用户态、Phase 2 内核模块

### 决策 2: 第一阶段是用户态守护进程（v11）
**理由：** Docker/K8s 证明不需要修改内核即可创造新的抽象层
**影响：** 降低入门门槛，MVP 可在 35 天内交付

### 决策 3: Capability 替代 ACL（v4, v13）
**理由：** Agent 的身份是动态的、能力需要委派/衰减
**影响：** 整个安全模型围绕 CapabilityCertificate 设计

### 决策 4: 认知公平 ≠ 计算公平（v2）
**理由：** 均等分配 LLM token 不是最优策略
**影响：** COI 模型、Meta-Scheduler 自适应

### 决策 5: 语义记忆 ≠ 文件系统（v3, v9）
**理由：** Agent 按内容寻址而非按路径
**影响：** S-Clock 置换、Cognode 数据结构

### 决策 6: Intent 替代 IPC（v2）
**理由：** Agent 间通信是自然语义而非结构化消息
**影响：** IntentMessage 格式、语义路由

### 决策 7: Rust 全栈（v5）
**理由：** 安全 + 性能 + 异步生态
**影响：** 7 crates、tokio、gRPC

### 决策 8: 混合模型策略（v14）
**理由：** 成本控制 + 延迟优化
**影响：** ModelRouter + 本地模型 fallback

### 决策 9: 纵深防御（v12, v15）
**理由：** 单一安全层不够——Agent 面临 Prompt Injection 等独特威胁
**影响：** 5 层 Prompt 防护 + 6 层沙箱隔离

### 决策 10: 渐进式内核集成（v16）
**理由：** 先用户态验证，再移入内核
**影响：** CognitiveSchedClass、eBPF 策略

---

## 三、实施路线图

```
═══════════════════════════════════════════════════════════════
                     NOW → LAK v1.0
═══════════════════════════════════════════════════════════════

Week 1-2:  项目初始化 + 核心类型  →  plan_detail.md Step 1-2
Week 3-4:  TAL (LLM + Tools)     →  plan_detail.md Step 3-4
Week 5-6:  ARE (Agent Runtime)   →  plan_detail.md Step 5-6
Week 7:    集成 + 端到端测试      →  plan_detail.md Step 7
Week 8:    安全加固 + 部署       →  plan_detail.md Step 8
───────────────────────────────────────────────────────────────
Month 3:   Phase 1 完成 (LAKd v0.1.0)
Month 4-6: Multi-Agent 协作 (v0.2.0)
Month 7-9: 高级记忆管理 (v0.3.0)
Month 10-12: 内核集成原型 (v0.5.0)
Year 2:    生产化 (v1.0.0)
═══════════════════════════════════════════════════════════════
```

---

## 四、最终系统验证清单

```
□ Agent 创建: 成功创建 Agent，返回 AgentId
□ 任务调度: CognitiveTask 被正确调度和执行
□ LLM 调用: Agent 可以调用 LLM 进行推理
□ 工具执行: Agent 可以在沙箱中执行工具
□ Capability 检查: 无能力的操作被正确拒绝
□ 意图路由: Agent 间可以发送和接收 Intent
□ 记忆存取: Agent 可以存取语义记忆
□ 错误恢复: Agent 从 LLM 错误中恢复
□ Prompt 注入防护: 已知注入模式被阻止
□ 沙箱隔离: 工具执行不逃逸沙箱
□ 审计日志: 所有关键操作有完整记录
□ 健康监控: 认知指标可观测
□ 备份恢复: 记忆和状态可从备份恢复
```

---

## 五、详细实施计划 → plan_detail.md

下一步：创建 `/project/windows_agent/plan_detail.md`，包含：
- Step-by-step 实现细节
- 每个 crate 的精确文件清单
- 每个模块的具体代码骨架
- 测试策略
- CI/CD 配置

# LAK — Linux Agent Kernel 智能体内核

> **"Not an OS for humans. An OS for agents."**

LAK (Linux Agent Kernel) is a userspace kernel for AI agents. It treats
LLM agents the way an operating system treats processes — with scheduling,
memory management, IPC, capability-based security, and crash recovery —
but the managed resource is *cognition* (tokens, reasoning, intents)
instead of CPU cycles and bytes.

Phase 1 ships as a gRPC daemon (`lakd`). Future phases move toward
multi-agent collaboration and Linux kernel integration
(`CognitiveSchedClass`, eBPF policy).

---

## Why an Agent Kernel?

| Traditional OS | LAK |
| --- | --- |
| Process / PID | Agent / `AgentId` |
| Thread | CognitiveTask (`TaskId`) |
| Scheduler (CFS) | CognitiveScheduler — COI-based, priority × aging |
| Virtual memory pages | MemoryChunk — semantic, content-addressed |
| Page replacement (clock) | S-Clock: recency + frequency + importance |
| IPC (pipes, signals) | IntentMessage — natural-language pub/sub |
| ACL / users | CapabilityCertificate — delegable, attenuating, expiring |
| Journal / WAL | CognitiveJournal — task state transitions |

Key design decisions (from 20 rounds of design iteration, see `plan.md`):

1. **Linux-first** — userspace first (like Docker), kernel integration later.
2. **Capability, not ACL** — agent identity is dynamic; rights must be
   delegable and attenuable.
3. **Cognitive fairness ≠ compute fairness** — scheduling optimizes for
   *opportunity* (COI), not equal time slices.
4. **Semantic memory ≠ filesystem** — agents address memories by content,
   not path, with tiered promotion (Working → ShortTerm → LongTerm → Archival).
5. **Hybrid models** — a router sends cheap tasks to free/local models and
   critical reasoning to cloud models.
6. **Defense in depth** — 5-layer prompt-injection defense + capability
   enforcement on every tool call + audit logging.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      lakd (gRPC :9191)                     │  ← clients / SDK
├────────────────────────────────────────────────────────────┤
│                    lak-services (kernel)                   │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ Scheduler   │ │ IntentRouter │ │ MemoryService       │  │
│  │ (COI +      │ │ (pub/sub +   │ │ (TF-IDF + S-Clock   │  │
│  │  aging)     │ │  dead-letter)│ │  + tier promotion)  │  │
│  └─────────────┘ └──────────────┘ └─────────────────────┘  │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ Reasoning   │ │ ToolRegistry │ │ CognitiveJournal    │  │
│  │ (5-stage    │ │ (+capability │ │ (WAL + checkpoints) │  │
│  │  pipeline)  │ │  enforcement)│ │                     │  │
│  └─────────────┘ └──────────────┘ └─────────────────────┘  │
│  ┌─────────────────────────┐ ┌───────────────────────────┐ │
│  │ InjectionDefense (5层)  │ │ SpeculativeEngine         │ │
│  └─────────────────────────┘ └───────────────────────────┘ │
├────────────────────────────────────────────────────────────┤
│  lak-are (AgentProcess runtime) │ lak-tal (LLM + tools)    │
│                                 │  OpenAI / Anthropic /    │
│                                 │  Ollama drivers,         │
│                                 │  sandboxed tools         │
├────────────────────────────────────────────────────────────┤
│  lak-core (types, AgentKernel trait) │ lak-proto (proto)   │
└────────────────────────────────────────────────────────────┘
```

### Crates

| Crate | Purpose |
| --- | --- |
| `lak-core` | Core types (Agent, CognitiveTask, Intent, MemoryChunk, Capability), `AgentKernel` trait, token budget |
| `lak-proto` | gRPC/Protobuf definitions (`proto/lak.proto`) |
| `lak-tal` | Tool Abstraction Layer: streaming LLM drivers (OpenAI, Anthropic, Ollama) and sandboxed tools (FileRead, ShellCmd, HttpGet) |
| `lak-are` | Agent Runtime Environment: `AgentProcess`, context window, COI stats |
| `lak-services` | The kernel itself: scheduler, intent router, semantic memory, reasoning pipeline, injection defense, journal, speculative engine |
| `lakd` | The daemon: gRPC server, driver registration, graceful shutdown |

---

## Quick Start

### Build

```bash
# requires: rustc ≥ 1.85, protoc
cargo build --workspace
```

### Run the daemon

```bash
# optional: configure LLM backends via environment
export ANTHROPIC_API_KEY=sk-...        # enables Anthropic driver
export OPENAI_API_KEY=sk-...           # enables OpenAI driver
export OLLAMA_URL=http://localhost:11434   # enables local Ollama driver

cargo run -p lakd
# [LAK] gRPC server listening on 0.0.0.0:9191
```

Configuration environment variables (see `config/lakd.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `LAK_LISTEN_ADDR` | `0.0.0.0:9191` | gRPC bind address |
| `LAK_MAX_AGENTS` | `1000` | Max concurrent agents |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4o` | Enable OpenAI driver |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-sonnet-5` | Enable Anthropic driver |
| `OLLAMA_URL` / `OLLAMA_MODEL` | — / `llama3.1` | Enable Ollama driver |
| `LAK_DISABLE_CLOUD_LLM` | unset | `1` = skip cloud drivers |
| `RUST_LOG` | `lak=debug,tonic=info` | Log filter |

### Docker

```bash
docker build -t lak:latest .
docker run -p 9191:9191 lak:latest
```

### Try it (grpcurl)

`examples/grpc_quickstart.sh` walks through the full API: create agent →
submit task → store/query memory → intents → capabilities → status.

---

## The gRPC API (lak.AgentKernel)

- **Agent lifecycle**: `CreateAgent`, `DestroyAgent`, `GetAgent`,
  `ListAgents`, `PauseAgent`, `ResumeAgent`
- **Cognitive tasks**: `SubmitTask`, `CancelTask`, `GetTask`
- **Intents (IPC)**: `SendIntent`, `AwaitIntent`
- **Semantic memory**: `StoreMemory`, `QueryMemory`, `ForgetMemory`
- **Capabilities**: `GrantCapability`, `RevokeCapability`,
  `DelegateCapability`, `GetCapabilities`
- **System**: `GetSystemStatus`, `Shutdown`

---

## Security Model

LAK implements **defense in depth** against the threats unique to agents
(prompt injection, tool abuse, capability escalation):

1. **Prompt hardening** — system prompt always first, untrusted content
   wrapped in source tags.
2. **I/O tagging** — every context token carries a `TokenSource`
   (SystemPrompt / UserInput / ToolOutput / FileContent …).
3. **Content filter** — pattern scan for instruction overrides,
   delimiter injection, exfiltration and destructive commands;
   suspicious input is sanitized, blocked or quarantined.
4. **Capability boundary** — every tool call is checked against the
   agent's *merged* capability certificate using the concrete resource
   (`file:///path`, URL) from the call parameters. Delegation can only
   attenuate, never expand; granting requires a delegatable source.
5. **Audit** — every tool execution (allowed, blocked or failed) is
   written to the audit log with agent id, parameters and outcome.

Tools additionally run under a sandbox policy: path whitelists,
size/timeout limits, network policy (`None` / `LocalhostOnly` /
`Allowlist` / `All`), scheme restrictions.

---

## Development

```bash
cargo test --workspace                  # unit + integration tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Design documents:

- `plan.md` — 20-round design summary, key decisions, roadmap
- `plan_detail.md` — step-by-step implementation blueprint

### Roadmap

- **v0.1 (Phase 1, now)** — userspace kernel: full gRPC pipeline,
  capability security, TF-IDF semantic memory, COI scheduler, journal
- **v0.2** — multi-agent collaboration (supervisor, negotiation, WFG deadlock handling)
- **v0.3** — advanced memory (embeddings, vector backends, consolidation)
- **v0.5** — kernel integration prototype (sched class, eBPF policy)
- **v1.0** — production hardening

## License

Apache-2.0 — see [LICENSE](LICENSE).

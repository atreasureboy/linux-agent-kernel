#!/usr/bin/env bash
# ── LAK gRPC quickstart ─────────────────────────────────────────────
# Walks through the whole AgentKernel API against a running lakd:
#   agent lifecycle → task → memory → intents → capabilities → status
#
# Requirements: grpcurl, jq, python3. A lakd listening on $LAK_ADDR.
#
# Usage:  ./examples/grpc_quickstart.sh

set -euo pipefail

LAK_ADDR="${LAK_ADDR:-127.0.0.1:9191}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_ARGS=(-import-path "$REPO_ROOT/crates/lak-proto/proto" -proto lak.proto)

grpc() { grpcurl -plaintext "${PROTO_ARGS[@]}" "$@"; }

# UUID bytes encoded as base64 (proto bytes fields)
new_id() {
  python3 -c 'import uuid,base64,sys;sys.stdout.write(base64.b64encode(uuid.uuid4().bytes).decode())'
}

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

step "System status"
grpc "$LAK_ADDR" lak.AgentKernel/GetSystemStatus '{}'

step "Create agent (with a delegatable FileRead capability)"
AGENT_ID=$(grpc "$LAK_ADDR" lak.AgentKernel/CreateAgent '{
  "spec": {
    "name": "demo-agent",
    "systemPrompt": "You are a careful system agent.",
    "model": "claude-sonnet-5",
    "maxContextTokens": 32768,
    "memoryQuotaBytes": 1073741824,
    "initialCapabilities": [{
      "capType": 1,
      "scope": "file:///**",
      "permissions": 33
    }]
  }
}' | jq -r .agentId)
echo "agent_id=$AGENT_ID"

step "Get agent"
grpc "$LAK_ADDR" lak.AgentKernel/GetAgent "{\"agentId\": \"$AGENT_ID\"}" | jq '{name, state}'

step "Submit cognitive task"
TASK_ID=$(new_id)
grpc "$LAK_ADDR" lak.AgentKernel/SubmitTask "{
  \"taskId\": \"$TASK_ID\",
  \"agentId\": \"$AGENT_ID\",
  \"task\": {
    \"taskId\": \"$TASK_ID\",
    \"agentId\": \"$AGENT_ID\",
    \"taskType\": 1,
    \"priority\": {\"urgency\": 40, \"importance\": 50, \"contextAffinity\": 50},
    \"state\": 1,
    \"content\": {\"naturalLanguage\": \"Summarize the kernel audit log\"}
  }
}" >/dev/null
grpc "$LAK_ADDR" lak.AgentKernel/GetTask "{\"taskId\": \"$TASK_ID\"}" | jq '{state}'

step "Store + query semantic memory"
CHUNK_ID=$(new_id)
grpc "$LAK_ADDR" lak.AgentKernel/StoreMemory "{
  \"agentId\": \"$AGENT_ID\",
  \"chunk\": {
    \"chunkId\": \"$CHUNK_ID\",
    \"agentId\": \"$AGENT_ID\",
    \"tier\": 1,
    \"content\": {\"rawText\": \"The deployment window is Saturday 02:00 UTC\"}
  }
}"
grpc "$LAK_ADDR" lak.AgentKernel/QueryMemory "{
  \"agentId\": \"$AGENT_ID\",
  \"query\": \"when do we deploy\",
  \"topK\": 3
}" | jq '.chunks[].content.rawText'

step "Intent pub/sub (broadcast + await)"
INTENT_ID=$(new_id)
grpc "$LAK_ADDR" lak.AgentKernel/SendIntent "{
  \"intent\": {
    \"intentId\": \"$INTENT_ID\",
    \"senderId\": \"$AGENT_ID\",
    \"target\": {\"targetType\": 1},
    \"intentType\": 3,
    \"content\": {\"naturalLanguage\": \"security scan finished\"}
  }
}" >/dev/null
grpc "$LAK_ADDR" lak.AgentKernel/AwaitIntent "{
  \"agentId\": \"$AGENT_ID\",
  \"subscription\": {\"topicFilters\": {\"t1\": \"security\"}}
}" | jq '.intent.content.naturalLanguage' || echo "(no matching intent yet)"

step "Capabilities"
grpc "$LAK_ADDR" lak.AgentKernel/GetCapabilities "{\"agentId\": \"$AGENT_ID\"}" \
  | jq '.certificate.capabilities'

step "List agents"
grpc "$LAK_ADDR" lak.AgentKernel/ListAgents '{}' | jq '.agents[] | {name, state}'

step "Destroy agent"
grpc "$LAK_ADDR" lak.AgentKernel/DestroyAgent "{\"agentId\": \"$AGENT_ID\"}"
echo "done."

//! HttpGet tool — make HTTP GET requests

use async_trait::async_trait;

use super::*;
use lak_core::types::capability::{CapabilityPermission, CapabilityRequirement, CapabilityType};

#[derive(Debug, Default)]
pub struct HttpGetTool;

#[async_trait]
impl Tool for HttpGetTool {
    fn name(&self) -> &str {
        "HttpGet"
    }

    fn description(&self) -> &str {
        "Make an HTTP GET request to a URL. Restricted by network policy."
    }

    fn danger_level(&self) -> DangerLevel {
        DangerLevel::Low
    }

    fn required_capability(&self) -> CapabilityRequirement {
        CapabilityRequirement {
            cap_type: CapabilityType::NetworkHttp,
            scope: "http*".into(),
            min_permissions: CapabilityPermission::READ,
        }
    }

    fn required_capability_for(&self, params: &serde_json::Value) -> CapabilityRequirement {
        let url = params["url"].as_str().unwrap_or_default();
        CapabilityRequirement {
            cap_type: CapabilityType::NetworkHttp,
            scope: url.to_string(),
            min_permissions: CapabilityPermission::READ,
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        context: &ToolContext,
    ) -> Result<ToolResult, ToolError> {
        let url_str = params["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams("url is required".into()))?;

        // URLs must parse — an unparseable URL must never slip past the
        // network policy checks below (fail closed).
        let parsed = url::Url::parse(url_str)
            .map_err(|e| ToolError::InvalidParams(format!("invalid URL: {e}")))?;

        // Only plain HTTP(S) — reject file://, gopher:// etc.
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(ToolError::AccessDenied(format!(
                "URL scheme '{}' is not allowed; only http/https",
                parsed.scheme()
            )));
        }

        // Network policy check
        match &context.sandbox.network_policy {
            NetworkPolicy::None => {
                return Err(ToolError::AccessDenied(
                    "Network access is disabled by sandbox policy".into(),
                ));
            }
            NetworkPolicy::LocalhostOnly => {
                let host = parsed.host_str().unwrap_or("");
                if host != "localhost" && host != "127.0.0.1" && host != "::1" {
                    return Err(ToolError::AccessDenied(
                        "Only localhost access is allowed".into(),
                    ));
                }
            }
            NetworkPolicy::Allowlist(allowed) => {
                let host = parsed.host_str().unwrap_or("");
                // Exact match or subdomain of an allowlisted domain.
                // Substring `contains` is unsafe: "example.com" would
                // accept "evil-example.com".
                let allowed = allowed.iter().any(|a| {
                    host == a.as_str() || host.ends_with(format!(".{}", a.as_str()).as_str())
                });
                if !allowed {
                    return Err(ToolError::AccessDenied(format!(
                        "Access to '{host}' is not in the allowlist"
                    )));
                }
            }
            NetworkPolicy::All => {} // No restriction
        }

        let client = reqwest::Client::builder()
            .timeout(context.sandbox.timeout)
            .build()
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let mut req = client.get(url_str);

        // Add custom headers if provided
        if let Some(headers) = params["headers"].as_object() {
            for (key, value) in headers {
                if let Some(val) = value.as_str() {
                    req = req.header(key.as_str(), val);
                }
            }
        }

        let response = req.send().await.map_err(|e| {
            if e.is_timeout() {
                ToolError::Timeout
            } else {
                ToolError::ExecutionError(e.to_string())
            }
        })?;

        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        let body_len = body.len();

        // Truncate large responses (at a UTF-8 char boundary)
        const MAX_BODY: usize = 500_000;
        let truncated = body_len > MAX_BODY;
        let display_body = if truncated {
            truncate_utf8(&body, MAX_BODY)
        } else {
            body
        };

        Ok(ToolResult {
            success: status >= 200 && status < 400,
            output: serde_json::json!({
                "url": url_str,
                "status": status,
                "body": display_body,
                "body_size_bytes": body_len,
                "truncated": truncated,
            }),
            audit_info: Some(AuditInfo {
                resource: url_str.to_string(),
                action: "GET".into(),
                bytes_transferred: body_len as u64,
            }),
        })
    }
}

//! ToolRegistry — registers and discovers tools

use std::collections::HashMap;
use std::sync::Arc;

use lak_tal::llm::ToolDefinition;
use lak_tal::tools::Tool;

/// Registry of all available tools
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a new tool
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// List all registered tools
    pub fn list(&self) -> Vec<&Arc<dyn Tool>> {
        self.tools.values().collect()
    }

    /// Check if a tool is registered
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Number of registered tools
    pub fn count(&self) -> usize {
        self.tools.len()
    }

    /// Build LLM tool definitions from the registry (for use in LLM requests)
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

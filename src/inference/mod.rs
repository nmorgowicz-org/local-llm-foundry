use serde::{Deserialize, Serialize};

pub mod backend;
pub mod capabilities;
pub mod launch;
pub mod llama_cpp;
pub mod llama_cpp_capabilities;
pub mod llama_cpp_tools;
pub mod metrics;
pub mod process_tree;
pub mod rapid_mlx;
pub mod supervisor;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    #[default]
    LlamaCpp,
    RapidMlx,
}

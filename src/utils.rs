use chrono::{DateTime, Local};
use dialoguer::console::Emoji;
use std::collections::HashMap;

use crate::chat_manager::{AssistantMetadata, ModelMetadata};

pub const CHAT_GPT_KNOWLEDGE_CUTOFF: &str = "September 2021";

/// <https://platform.openai.com/docs/models/gpt-3-5>
pub const GPT_3_5_MODEL_NAME: &str = "gpt-3.5-turbo";
pub const GPT_3_5_MODEL_TOKEN_LIMIT: u32 = 4096;

/// <https://platform.openai.com/docs/models/gpt-4>
pub const GPT_4_8K_MODEL_NAME: &str = "gpt-4";
pub const GPT_4_8K_MODEL_TOKEN_LIMIT: u32 = 8192;

pub const GPT_4_32K_MODEL_NAME: &str = "gpt-4-32k";
pub const GPT_4_32K_MODEL_TOKEN_LIMIT: u32 = 32768;

// Emojis
pub const ROBOT_EMOJI: Emoji = Emoji("🤖", "");
pub const QUESTION_MARK_EMOJI: Emoji = Emoji("❓", "");
pub const SYSTEM_EMOJI: Emoji = Emoji("ℹ️ ", "");
pub const INCREASING_TREND_EMOJI: Emoji = Emoji("📈", "");

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq, Hash)]
pub enum ChatGptModel {
    #[default]
    GPT_3_5,
    GPT_4_8k,
    GPT_4_32k,
}

impl std::fmt::Display for ChatGptModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.get_model_name())
    }
}

impl ChatGptModel {
    pub fn get_model_name(&self) -> &str {
        match self {
            ChatGptModel::GPT_3_5 => GPT_3_5_MODEL_NAME,
            ChatGptModel::GPT_4_8k => GPT_4_8K_MODEL_NAME,
            ChatGptModel::GPT_4_32k => GPT_4_32K_MODEL_NAME,
        }
    }

    pub fn get_model_token_limit(&self) -> u32 {
        match self {
            ChatGptModel::GPT_3_5 => GPT_3_5_MODEL_TOKEN_LIMIT,
            ChatGptModel::GPT_4_8k => GPT_4_8K_MODEL_TOKEN_LIMIT,
            ChatGptModel::GPT_4_32k => GPT_4_32K_MODEL_TOKEN_LIMIT,
        }
    }

    pub fn to_model_metadata(self) -> ModelMetadata {
        ModelMetadata {
            name: self.get_model_name().to_owned(),
            token_limit: self.get_model_token_limit(),
        }
    }
}

pub fn now() -> DateTime<Local> {
    Local::now()
}

pub fn now_rfc3339() -> String {
    now().to_rfc3339()
}

pub const DEFAULT_SYSTEM_INSTRUCTIONS_KEY: &str = "default";

pub fn generate_system_instructions() -> HashMap<&'static str, AssistantMetadata> {
    let mut instructions = HashMap::new();

    let current_time_str = now_rfc3339();

    instructions.insert(
        DEFAULT_SYSTEM_INSTRUCTIONS_KEY,
        AssistantMetadata::new(format!(
            "You are ChatGPT, a large language model trained by OpenAI. 
Answer as concisely as possible. Knowledge cutoff year {} Current date and time: {}",
            CHAT_GPT_KNOWLEDGE_CUTOFF, current_time_str
        )),
    );

    instructions.insert(
        "joi",
        AssistantMetadata::new(format!(
            "You are Joi. The cheerful and helpful AI assistant. Answer as concisely as possible.
Knowledge cutoff year {} Current date and time: {}",
            CHAT_GPT_KNOWLEDGE_CUTOFF, current_time_str
        )),
    );

    instructions
}

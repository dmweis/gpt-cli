use crate::{
    configuration::get_project_dirs,
    utils::{
        CHAT_GPT_MODEL_NAME, CHAT_GPT_MODEL_TOKEN_LIMIT, INCREASING_TREND_EMOJI,
        QUESTION_MARK_EMOJI, ROBOT_EMOJI, SYSTEM_EMOJI,
    },
};
use anyhow::{Context, Result};
use async_openai::{
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageArgs,
        CreateChatCompletionRequestArgs, Role, Usage,
    },
    Client,
};
use chrono::prelude::{DateTime, Local};
use dialoguer::console::Term;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tiktoken_rs::cl100k_base;

/// Manager for conversations
pub struct ChatHistory {
    history: Vec<ChatCompletionRequestMessage>,
    token_usage: Option<Usage>,
    conversation_start: Option<DateTime<Local>>,
    conversation_title: Option<String>,
}

impl ChatHistory {
    pub fn new(prompt: &str) -> anyhow::Result<Self> {
        let history = vec![ChatCompletionRequestMessageArgs::default()
            .content(prompt)
            .role(Role::System)
            .build()?];
        let dt: DateTime<Local> = Local::now();
        Ok(Self {
            history,
            token_usage: None,
            conversation_start: Some(dt),
            conversation_title: None,
        })
    }

    /// Get Usage as reported by the API
    ///
    /// Usage is not reported in streaming mode for some reason
    pub fn token_usage(&self) -> Option<Usage> {
        self.token_usage.clone()
    }

    /// Use local tokenizer library to estimate token usage
    ///
    /// This can be imprecise if we have different tokenization rules than the model
    pub fn count_tokens(&self) -> i64 {
        // based on this https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        // but there some weird hacks because the counts weren't lining up

        // used by gpt-3.5-turbo-0301
        let bpe = cl100k_base().expect("Failed to load cl100k_base");
        // Start with -1 because somehow we always had 1 extra token
        let mut token_count = -1_i64;
        for message in &self.history {
            // each message adds 4 tokens
            // because every message follows <im_start>{role/name}\n{content}<im_end>\n
            match message.role {
                Role::User => {
                    token_count += 4;
                    if let Some(name) = &message.name {
                        // example says "if there's a name, the role is omitted"
                        // but it says "role is always required and always 1 token"
                        // so I don't know
                        token_count -= 1;
                        // add name to count
                        token_count += bpe.encode_with_special_tokens(name).len() as i64;
                    }
                }
                Role::System => {
                    token_count += 4;
                }
                Role::Assistant => {
                    // Assistant messages should be primed with <im_start>assistant
                    // so that'd be 2. But from my testing it looks like there are still 4
                    token_count += 4;
                }
            }

            // add role to count
            token_count += bpe
                .encode_with_special_tokens(&message.role.to_string())
                .len() as i64;

            // add message to count
            token_count += bpe.encode_with_special_tokens(&message.content).len() as i64;
        }
        token_count
    }

    /// fun attempt at generating titles for chats
    /// would be great if this could be async
    async fn populate_title_if_empty(&mut self, client: &Client) -> Result<()> {
        if self.conversation_title.is_none() {
            self.populate_title(client).await?;
        }
        Ok(())
    }

    /// create a new title for the chat using special ChatGPT query
    pub async fn populate_title(&mut self, client: &Client) -> Result<()> {
        let mut history_copy = self.history.clone();
        let message =
                "How would you title this conversation up until before this message? Answer in all lowercase with underscores 
\"_\" between words so that it can be used as a file name. Be concise.";

        let user_message = ChatCompletionRequestMessageArgs::default()
            .content(message)
            .role(Role::User)
            .build()?;

        history_copy.push(user_message);

        let request = CreateChatCompletionRequestArgs::default()
            .model(CHAT_GPT_MODEL_NAME)
            .messages(history_copy)
            .build()?;

        let response = client.chat().create(request).await?;

        let title = response.choices[0].message.content.trim().to_owned();
        self.conversation_title = Some(title);
        Ok(())
    }

    /// pop and return the last message in history
    pub fn pop_last_message(&mut self) -> Option<ChatCompletionRequestMessage> {
        self.history.pop()
    }

    /// generate next message
    pub async fn next_message(
        &mut self,
        user_message: &str,
        client: &Client,
    ) -> anyhow::Result<String> {
        let user_message = ChatCompletionRequestMessageArgs::default()
            .content(user_message)
            .role(Role::User)
            .build()?;

        self.history.push(user_message);

        let request = CreateChatCompletionRequestArgs::default()
            .model(CHAT_GPT_MODEL_NAME)
            .messages(self.history.clone())
            .build()?;

        let response = client.chat().create(request).await?;

        let added_response = ChatCompletionRequestMessageArgs::default()
            .content(response.choices[0].message.content.clone())
            .role(response.choices[0].message.role.clone())
            .build()?;

        self.history.push(added_response);
        self.token_usage = response.usage;

        self.populate_title_if_empty(client).await?;

        Ok(response.choices[0].message.content.clone())
    }

    /// stream next message to terminal
    pub async fn next_message_stream_stdout(
        &mut self,
        user_message: &str,
        client: &Client,
        term: &Term,
    ) -> anyhow::Result<String> {
        // this probably shouldn't leak abstraction to terminal
        // but until I have a use case where the abstriction helps this is okay....ish
        let user_message = ChatCompletionRequestMessageArgs::default()
            .content(user_message)
            .role(Role::User)
            .build()?;

        self.history.push(user_message);

        let request = CreateChatCompletionRequestArgs::default()
            .model(CHAT_GPT_MODEL_NAME)
            .messages(self.history.clone())
            .build()?;

        let mut stream = client.chat().create_stream(request).await?;

        let mut response_role = None;
        let mut response_content_buffer = String::new();

        term.hide_cursor()?;

        // For reasons not documented in OpenAI docs / OpenAPI spec, the response of streaming call is different and doesn't include all the same fields.
        while let Some(result) = stream.next().await {
            let response = result?;
            if let Some(new_usage) = response.usage {
                self.token_usage = Some(new_usage);
            }

            // this ignores if there are multiple choices on the answer
            let delta = &response
                .choices
                .first()
                .context("No first choice on response")?
                .delta;
            // role and content are not guaranteed to be set on all deltas

            if let Some(role) = &delta.role {
                response_role = Some(role.clone());
            }

            if let Some(delta_content) = &delta.content {
                response_content_buffer.push_str(delta_content);
                term.write_str(delta_content)?;
            }
        }

        // this markdown thing doesn't work very well right now
        // re-render as markdown
        // consider adding slowdown for effect?
        // let lines = response_content_buffer.lines().count();
        // term.clear_last_lines(lines)?;
        // // this markdown is weird. Doesn't render correctly I think
        // let markdown = termimad::inline(&response_content_buffer);
        // term.write_line(&format!("{}", markdown))?;

        // empty new line after stream is done

        term.write_line("\n")?;

        // print usage recorded
        if let Some(token_usage) = self.token_usage.as_ref() {
            let msg = format!("{INCREASING_TREND_EMOJI} Recorded usage {}/{CHAT_GPT_MODEL_TOKEN_LIMIT} tokens used",token_usage.total_tokens);
            term.write_line(&msg)?;
        }

        // print usage calculated
        let usage_msg = format!(
            "{INCREASING_TREND_EMOJI} Estimated usage {}/{CHAT_GPT_MODEL_TOKEN_LIMIT} tokens used",
            self.count_tokens()
        );
        term.write_line(&usage_msg)?;

        term.show_cursor()?;

        let added_response = ChatCompletionRequestMessageArgs::default()
            .content(&response_content_buffer)
            .role(response_role.unwrap_or(Role::Assistant))
            .build()?;

        self.history.push(added_response);

        self.populate_title_if_empty(client).await?;

        if let Some(title) = &self.conversation_title {
            term.set_title(title.replace('_', " "));
        }

        Ok(response_content_buffer)
    }

    /// print history of chat to terminal
    pub fn print_history(&self, term: &Term) -> Result<()> {
        // this should probably not live here
        term.write_line("---------------------------------")?;
        term.write_line("Conversation so far:")?;
        for message in &self.history {
            match message.role {
                Role::System => term.write_line(&format!("{SYSTEM_EMOJI} System:\n"))?,
                Role::Assistant => term.write_line(&format!("{ROBOT_EMOJI} ChatGPT:\n"))?,
                Role::User => term.write_line(&format!("{QUESTION_MARK_EMOJI} User:\n"))?,
            }
            term.write_line(&message.content)?;
        }

        term.write_line("")?;
        // print usage recorded
        if let Some(token_usage) = self.token_usage.as_ref() {
            term.write_line(&format!(
                "{INCREASING_TREND_EMOJI} Recorded usage {}/{CHAT_GPT_MODEL_TOKEN_LIMIT} tokens",
                token_usage.total_tokens
            ))?;
        }

        // print usage calculated
        term.write_line(&format!(
            "{INCREASING_TREND_EMOJI} Estimated usage {}/{CHAT_GPT_MODEL_TOKEN_LIMIT} tokens",
            self.count_tokens()
        ))?;
        term.write_line("---------------------------------")?;
        Ok(())
    }

    /// save chat history file
    pub fn save_to_file(&self) -> Result<()> {
        // TODO(David): Extract this outside
        let project_dirs = get_project_dirs()?;
        let cache_dir = project_dirs.cache_dir();

        std::fs::create_dir_all(cache_dir).context("failed to crate user cache directory")?;

        let time = self
            .conversation_start
            .unwrap_or_else(Local::now)
            .to_rfc3339();

        let title = self
            .conversation_title
            .clone()
            .map(|title| format!("{title}_"))
            .unwrap_or_default();

        let file_path = cache_dir.join(format!("{title}{time}.yaml"));

        let history_for_storage: Vec<ChatHistoryElement> =
            self.history.iter().map(|element| element.into()).collect();

        let history_storage = ChatHistoryStorage {
            messages: history_for_storage,
        };

        let file = std::fs::File::create(file_path)?;
        serde_yaml::to_writer(file, &history_storage)?;
        Ok(())
    }

    pub fn get_all_saved_conversations() -> Result<Vec<PathBuf>> {
        let project_dirs = get_project_dirs()?;
        let cache_dir = project_dirs.cache_dir();

        let mut files = vec![];

        for entry in std::fs::read_dir(cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            }
        }
        Ok(files)
    }

    /// load from chat history file
    pub fn load_from_file(file_path: &Path) -> anyhow::Result<ChatHistory> {
        let file = std::fs::File::open(file_path)?;
        let chat_history: ChatHistoryStorage = serde_yaml::from_reader(file)?;
        let converted_history_list: Vec<ChatCompletionRequestMessage> = chat_history
            .messages
            .into_iter()
            .map(|message| message.into())
            .collect();
        Ok(ChatHistory {
            history: converted_history_list,
            token_usage: None,
            conversation_start: None,
            conversation_title: None,
        })
    }
}

impl From<&ChatCompletionRequestMessage> for ChatHistoryElement {
    fn from(source: &ChatCompletionRequestMessage) -> Self {
        Self {
            role: source.role.clone(),
            content: source.content.clone(),
            name: source.name.clone(),
        }
    }
}

impl From<ChatHistoryElement> for ChatCompletionRequestMessage {
    fn from(source: ChatHistoryElement) -> Self {
        Self {
            role: source.role,
            content: source.content,
            name: source.name,
        }
    }
}

/// used for storage
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatHistoryStorage {
    /// message
    pub messages: Vec<ChatHistoryElement>,
}

/// Used for storage because [ChatCompletionRequestMessage] is not fully serde'd
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatHistoryElement {
    /// The role of the author of this message.
    pub role: Role,
    /// The contents of the message
    pub content: String,
    /// The name of the user in a multi-user chat
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

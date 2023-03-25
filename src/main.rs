mod chat_manager;
mod cli_history;
mod configuration;
mod utils;

use anyhow::Context;
use async_openai::Client;
use clap::Parser;
use cli_history::InMemoryHistory;
use configuration::AppConfig;
use dialoguer::{console::Term, theme::ColorfulTheme, FuzzySelect, Input};
use std::path::PathBuf;
use utils::{
    generate_system_instructions, CHAT_GPT_MODEL_TOKEN_LIMIT, INCREASING_TREND_EMOJI, ROBOT_EMOJI,
};

#[derive(Parser)]
#[command()]
struct Cli {
    /// load from file
    #[arg(long)]
    file: Option<PathBuf>,
    /// list files
    #[arg(long)]
    select_file: bool,
    /// save conversation history
    #[arg(long, default_value_t = true)]
    save: bool,
    /// save default config and exit
    #[arg(long)]
    create_config: bool,
    /// stream
    #[arg(long, default_value_t = true)]
    stream: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    if cli.create_config {
        // write default config
        let config_new = AppConfig::default();
        config_new.save_user_config()?;
        return Ok(());
    }

    let term = Term::stdout();
    let mut history = InMemoryHistory::default();
    let term_theme = ColorfulTheme::default();

    if cli.select_file {
        let files = chat_manager::ChatHistory::get_all_saved_conversations()?;
        let file_names: Vec<_> = files
            .iter()
            .map(|path| {
                path.file_name()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap_or_default()
            })
            .collect();
        let selection = FuzzySelect::with_theme(&term_theme)
            .with_prompt("Select file")
            .items(&file_names)
            .default(0)
            .interact_on(&term)?;
        cli.file = Some(
            files
                .get(selection)
                .context("Selected wrong item form file list")?
                .to_owned(),
        );
        // weird mutating the cli args
    }

    let config = AppConfig::load_user_config()?;

    let client = Client::new().with_api_key(&config.open_ai_api_key);

    let system_messages = generate_system_instructions();

    let mut chat_manager = if let Some(path) = cli.file {
        chat_manager::ChatHistory::load_from_file(&path)?
    } else {
        chat_manager::ChatHistory::new(&system_messages["joi"])?
    };

    loop {
        let mut user_question: String = Input::with_theme(&term_theme)
            .with_prompt("Question:")
            .history_with(&mut history)
            .interact_text_on(&term)?;

        if &user_question == "/?" {
            let options = UserActions::all_str();

            let selection = FuzzySelect::with_theme(&term_theme)
                .with_prompt("Select action")
                .items(&options)
                .default(0)
                .interact_on_opt(&term)?;
            let selection = selection.and_then(|index| UserActions::all().get(index));
            match selection {
                Some(UserActions::ReturnToChat) => continue,
                Some(UserActions::RecreateTitle) => {
                    chat_manager.populate_title(&client).await?;
                    continue;
                }
                Some(UserActions::RegenerateResponse) => {
                    // ugly...
                    _ = chat_manager.pop_last_message();
                    user_question = chat_manager.pop_last_message().unwrap_or_default().content;
                    // keep going to create new message
                }
                Some(UserActions::PrintChatHistory) => {
                    chat_manager.print_history(&term)?;
                    continue;
                }
                None => continue,
            }
        }

        term.write_line(&format!("\n{ROBOT_EMOJI} ChatGPT:\n"))?;

        if cli.stream {
            let _response = chat_manager
                .next_message_stream_stdout(&user_question, &client, &term)
                .await?;
        } else {
            let response = chat_manager.next_message(&user_question, &client).await?;

            term.write_line(&response)?;

            // print usage
            if let Some(token_usage) = chat_manager.token_usage() {
                term.write_line(&format!(
                    "\n{INCREASING_TREND_EMOJI} Recorded usage {}/{CHAT_GPT_MODEL_TOKEN_LIMIT} tokens used",
                    token_usage.total_tokens
                ))?;
            }

            // print usage calculated
            term.write_line(&format!(
                "{INCREASING_TREND_EMOJI} Estimated usage {}/{CHAT_GPT_MODEL_TOKEN_LIMIT} tokens used",
                chat_manager.count_tokens()
            ))?;

            term.write_line("")?;
        }

        if cli.save {
            chat_manager.save_to_file()?;
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum UserActions {
    ReturnToChat,
    RecreateTitle,
    RegenerateResponse,
    PrintChatHistory,
}

impl UserActions {
    fn as_str(&self) -> &'static str {
        match self {
            UserActions::ReturnToChat => "Return to chat",
            UserActions::RecreateTitle => "Recreate title",
            UserActions::RegenerateResponse => "Regenerate response",
            UserActions::PrintChatHistory => "Print chat history",
        }
    }

    fn all() -> &'static [UserActions] {
        &[
            UserActions::ReturnToChat,
            UserActions::RecreateTitle,
            UserActions::RegenerateResponse,
            UserActions::PrintChatHistory,
        ]
    }

    fn all_str() -> Vec<&'static str> {
        Self::all().iter().map(|opt| opt.as_str()).collect()
    }
}

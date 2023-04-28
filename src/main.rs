mod chat_manager;
mod cli_history;
mod configuration;
mod utils;

use anyhow::Context;
use async_openai::Client;
use clap::{Parser, Subcommand};
use cli_history::InMemoryHistory;
use configuration::{AppConfig, OPEN_AI_API_KEY_WEB_URL};
use dialoguer::{console::Term, theme::ColorfulTheme, FuzzySelect, Input, Password};
use std::path::PathBuf;
use utils::{
    generate_system_instructions, ChatGptModel, DEFAULT_SYSTEM_INSTRUCTIONS_KEY, ROBOT_EMOJI,
};

#[derive(Parser)]
#[command()]
struct Cli {
    /// model to select
    #[arg(long, value_enum, default_value = "gpt-3-5")]
    model: ChatGptModel,
    /// load from file
    #[arg(long)]
    file: Option<PathBuf>,
    /// list files
    #[arg(long)]
    select_file: bool,
    /// don't save conversation history
    #[arg(long)]
    no_save: bool,
    /// disable streaming
    #[arg(long)]
    no_stream: bool,

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    #[arg(long)]
    temperature: Option<f32>,

    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// min: 0, max: 1, default: 1
    ///  We generally recommend altering this or `temperature` but not both.
    #[arg(long)]
    top_p: Option<f32>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// login
    Login,
    /// create default config
    CreateConfig,
}

// #[derive(Args)]
// struct LoginArgs {}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    let term = Term::stdout();
    let mut history = InMemoryHistory::default();
    let term_theme = ColorfulTheme::default();

    match cli.command {
        Some(Commands::Login) => {
            term.write_line(&format!("Get token from {OPEN_AI_API_KEY_WEB_URL}"))?;
            let api_key: String = Password::with_theme(&term_theme)
                .with_prompt("API key:")
                .interact_on(&term)?;
            let config = AppConfig::new(api_key);
            config.save_user_config()?;
            term.write_line("Login successful")?;
            return Ok(());
        }
        Some(Commands::CreateConfig) => {
            // write default config
            let config_new = AppConfig::default();
            config_new.save_user_config()?;
            return Ok(());
        }
        None => {}
    }

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
        chat_manager::ChatHistory::new(
            cli.model.to_model_metadata(),
            system_messages[DEFAULT_SYSTEM_INSTRUCTIONS_KEY].clone(),
        )?
    };

    term.write_line("Write /? to get help")?;

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

        if !cli.no_stream {
            let _response = chat_manager
                .next_message_stream_stdout(
                    &user_question,
                    &client,
                    &term,
                    cli.temperature,
                    cli.top_p,
                )
                .await?;
        } else {
            let response = chat_manager
                .next_message(&user_question, &client, cli.temperature, cli.top_p)
                .await?;

            term.write_line(&response)?;
            term.write_line("")?;
            // print usage
            if let Some(token_usage) = chat_manager.token_usage_message() {
                term.write_line(&token_usage)?;
            }
            term.write_line(&chat_manager.token_count_message())?;
            term.write_line("")?;
        }

        if !cli.no_save {
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

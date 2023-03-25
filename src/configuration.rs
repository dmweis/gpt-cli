use anyhow::{Context, Result};
use config::Config;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const PROJECT_QUALIFIER: &str = "com";
const PROJECT_ORGANIZATION: &str = "dmweis";
const PROJECT_APPLICATION_NAME: &str = "gpt-cli";

const GPT_CLI_CONFIG_FILE_NAME: &str = "config";
const GPT_CLI_CONFIG_FILE_EXTENSION: &str = "yaml";

pub fn get_project_dirs() -> Result<ProjectDirs> {
    ProjectDirs::from(
        PROJECT_QUALIFIER,
        PROJECT_ORGANIZATION,
        PROJECT_APPLICATION_NAME,
    )
    .context("failed to establish project dirs")
}

fn get_config_file_path() -> Result<PathBuf> {
    let proj_dirs = get_project_dirs()?;
    let config_dir_path = proj_dirs.config_dir();
    Ok(config_dir_path.join(GPT_CLI_CONFIG_FILE_NAME))
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AppConfig {
    pub open_ai_api_key: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            open_ai_api_key: String::from(
                "Get token from https://platform.openai.com/account/api-keys",
            ),
        }
    }
}

impl AppConfig {
    pub fn load_user_config() -> anyhow::Result<Self> {
        let config_file_path = get_config_file_path()?;
        let settings = Config::builder()
            .add_source(config::File::from(config_file_path))
            .add_source(config::Environment::with_prefix("GPT"))
            .build()?;

        Ok(settings.try_deserialize::<AppConfig>()?)
    }

    pub fn save_user_config(&self) -> anyhow::Result<()> {
        let config_file_path =
            get_config_file_path()?.with_extension(GPT_CLI_CONFIG_FILE_EXTENSION);

        std::fs::create_dir_all(
            config_file_path
                .parent()
                .context("failed to get config file parent directory")?,
        )?;

        let file = std::fs::File::create(config_file_path)?;
        serde_yaml::to_writer(file, self)?;
        Ok(())
    }
}

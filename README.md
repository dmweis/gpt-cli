# GPT-CLI

[![codecov](https://codecov.io/gh/dmweis/gpt-cli/branch/main/graph/badge.svg)](https://codecov.io/gh/dmweis/gpt-cli)
[![Rust](https://github.com/dmweis/gpt-cli/workflows/Rust/badge.svg)](https://github.com/dmweis/gpt-cli/actions)
[![Private docs](https://github.com/dmweis/gpt-cli/workflows/Deploy%20Docs%20to%20GitHub%20Pages/badge.svg)](https://davidweis.dev/gpt-cli/gpt-cli/index.html)

Rust CLI for chat gpt

## API key

Get key from [OpenAI account](https://platform.openai.com/account/api-keys)


## ChatGPT cli

`cargo run --bin gpt-cli` to run cli (it's also the default target for `cargo run`)

non-exhaustive list of features:

* read user config
* save previous conversations
* title conversations using generated summary titles

### Installation with cargo

```bash
cargo install --git https://github.com/dmweis/gpt-cli
```

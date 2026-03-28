# ForgeTune

ForgeTune is a self-hosted, Dockerized utility for building and managing LLM fine-tuning datasets. It ships as a single FastAPI application with a browser-based GUI, persistent SQLite storage, dataset import/export tooling, curation utilities, OpenAI-compatible LLM assistance, and built-in fine-tuning job orchestration.

## Features

- Dataset CRUD with schema-aware defaults for Alpaca, ShareGPT, and OpenAI chat-style records.
- Manual example entry, bulk upload for JSON, JSONL, CSV, and Parquet, and synthetic example generation through Ollama, OpenAI, or OpenAI-compatible local APIs such as vLLM.
- Saved provider profiles for OpenAI, OpenAI-compatible gateways, and Ollama, including reusable connection settings and model discovery.
- External acquisition through SearxNG result import, same-domain web crawling, and GitHub repository, code, or issue search.
- Curation workflow for whitespace cleanup, empty-row removal, deduplication by content hash, and token estimation using `tiktoken`.
- Annotation workspace for reviewing, editing, deleting, labeling, and LLM-assisted refinement of individual instruction-output examples.
- Built-in OpenAI-compatible fine-tuning flow that exports chat JSONL, uploads the training file, starts the job, and tracks the remote run locally.
- Export support for Alpaca, ShareGPT, OpenAI chat JSONL, Axolotl, Unsloth, and Hugging Face-compatible message JSONL.

## Run with Docker

Create a local env file first:

```bash
cp .env.example .env
```

```bash
docker compose up --build -d
```

Open http://localhost:8000.

The compose stack now brings up four services together:

- ForgeTune on `http://localhost:8000`
- A bundled SearxNG instance on `http://localhost:8080`
- An internal crawler service that handles SearxNG and web-acquisition requests for the main app
- A Valkey cache backing the SearxNG container

By default, the ForgeTune container points its SearxNG integration at the bundled `searxng` service and forwards web-crawl and SearxNG imports to the internal `crawler` service. If you want synthetic generation to hit Ollama running on the Docker host, leave `FORGETUNE_OLLAMA_BASE_URL` at its default `http://host.docker.internal:11434` value. If you want the backend to have default OpenAI or GitHub settings, put them in `.env` before starting the stack.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`app.main` loads `.env` automatically, so the same file works for local development and Docker Compose.

## Environment variables

Use `.env.example` as the starting point for `.env`. The most useful settings are:

- `FORGETUNE_OPENAI_BASE_URL`, `FORGETUNE_OPENAI_API_KEY`, `FORGETUNE_OPENAI_ORGANIZATION`, `FORGETUNE_OPENAI_PROJECT`, `FORGETUNE_OPENAI_MODEL`
- `FORGETUNE_OPENAI_COMPATIBLE_BASE_URL`, `FORGETUNE_OPENAI_COMPATIBLE_API_KEY`, `FORGETUNE_OPENAI_COMPATIBLE_ORGANIZATION`, `FORGETUNE_OPENAI_COMPATIBLE_PROJECT`, `FORGETUNE_OPENAI_COMPATIBLE_MODEL`
- `FORGETUNE_OLLAMA_BASE_URL`, `FORGETUNE_OLLAMA_MODEL`
- `FORGETUNE_GITHUB_BASE_URL`, `FORGETUNE_GITHUB_TOKEN`, `FORGETUNE_GITHUB_REPOSITORY`
- `FORGETUNE_SEARXNG_BASE_URL`
- `FORGETUNE_CRAWLER_SERVICE_URL`

These values are used as startup defaults for the UI and backend. Secrets such as API keys and GitHub tokens are read server-side and used as fallbacks when the corresponding form field is left blank.

## LLM providers and assistance

- Saved provider profiles live in the Operate tab and can target `openai`, `openai-compatible`, or `ollama` endpoints.
- OpenAI: use `https://api.openai.com` and either store the API key in a saved provider profile or set `FORGETUNE_OPENAI_API_KEY` in `.env`.
- OpenAI-compatible APIs: use the `FORGETUNE_OPENAI_COMPATIBLE_*` variables when you want different defaults for gateways such as vLLM, LiteLLM, OpenRouter, or GitHub Models.
- Ollama: use `http://host.docker.internal:11434` from inside Docker.
- vLLM or other OpenAI-compatible APIs: point the base URL to the server root, for example `http://host.docker.internal:8001` if that service exposes `/v1/chat/completions`, `/v1/models`, `/v1/files`, and `/v1/fine_tuning/jobs`.
- The Capture tab can use a saved provider profile or an ad hoc authenticated request for synthetic example generation.
- The Review tab can call a saved provider profile to draft a missing output or improve an existing example in place.

## Built-in fine-tuning

- Fine-tuning currently targets providers that implement the OpenAI-style `/v1/files` and `/v1/fine_tuning/jobs` APIs.
- The app exports the selected dataset in OpenAI chat JSONL, uploads it with `purpose=fine-tune`, starts the remote job, and stores the returned file and job IDs in the local SQLite database.
- Fine-tune jobs remain visible after a restart and can be synced or canceled later from the Operate tab.
- Ollama is supported for generation and assistance, but not for fine-tuning because it does not expose an OpenAI fine-tuning API.

## External source integrations

- SearxNG: the default base URL is now the bundled compose service at `http://searxng:8080` from inside ForgeTune, and `http://localhost:8080` from your browser if you want to inspect the search UI directly.
- Web crawl: seed one or more URLs, keep the crawl same-domain by default, and import the extracted page text as draft examples for later annotation.
- GitHub: search repositories, code, or issues via the GitHub REST API. Add a personal access token if you need higher rate limits or private-resource access. You can also set `FORGETUNE_GITHUB_TOKEN` in `.env` so GitHub imports use it automatically when the form token field is empty.
- If an internal service or intercepted network path presents a non-standard TLS certificate, you can disable HTTPS verification for that specific import from the UI. Leave verification enabled for normal public endpoints.

## Notes

- Data is stored in `./data/dataset_manager.db`.
- The SearxNG runtime config is stored in `./searxng/settings.yml`; replace the development `secret_key` before exposing the service outside your machine.
- Uploaded records are normalized into a canonical internal schema and converted back out on export.
- Export formats intentionally target the common data layouts consumed by Axolotl, Unsloth, and Hugging Face training pipelines.
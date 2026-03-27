# ForgeTune

ForgeTune is a self-hosted, Dockerized utility for building and managing LLM fine-tuning datasets. It ships as a single FastAPI application with a browser-based GUI, persistent SQLite storage, dataset import/export tooling, curation utilities, and local LLM-powered synthetic data generation.

## Features

- Dataset CRUD with schema-aware defaults for Alpaca, ShareGPT, and OpenAI chat-style records.
- Manual example entry, bulk upload for JSON, JSONL, CSV, and Parquet, and synthetic example generation through Ollama or OpenAI-compatible local APIs such as vLLM.
- External acquisition through SearxNG result import, same-domain web crawling, and GitHub repository, code, or issue search.
- Curation workflow for whitespace cleanup, empty-row removal, deduplication by content hash, and token estimation using `tiktoken`.
- Annotation workspace for reviewing, editing, deleting, and labeling individual instruction-output examples.
- Export support for Alpaca, ShareGPT, OpenAI chat JSONL, Axolotl, Unsloth, and Hugging Face-compatible message JSONL.

## Run with Docker

```bash
docker compose up --build -d
```

Open http://localhost:8000.

The compose stack now brings up four services together:

- ForgeTune on `http://localhost:8000`
- A bundled SearxNG instance on `http://localhost:8080`
- An internal crawler service that handles SearxNG and web-acquisition requests for the main app
- A Valkey cache backing the SearxNG container

By default, the ForgeTune container points its SearxNG integration at the bundled `searxng` service and forwards web-crawl and SearxNG imports to the internal `crawler` service. If you want synthetic generation to hit Ollama running on the Docker host, leave `FORGETUNE_OLLAMA_BASE_URL` at its default `http://host.docker.internal:11434` value.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Synthetic data providers

- Ollama: use `http://host.docker.internal:11434` from inside Docker.
- vLLM or other OpenAI-compatible APIs: point the base URL to the server root, for example `http://host.docker.internal:8001` if that service exposes `/v1/chat/completions`.

## External source integrations

- SearxNG: the default base URL is now the bundled compose service at `http://searxng:8080` from inside ForgeTune, and `http://localhost:8080` from your browser if you want to inspect the search UI directly.
- Web crawl: seed one or more URLs, keep the crawl same-domain by default, and import the extracted page text as draft examples for later annotation.
- GitHub: search repositories, code, or issues via the GitHub REST API. Add a personal access token if you need higher rate limits or private-resource access.
- If an internal service or intercepted network path presents a non-standard TLS certificate, you can disable HTTPS verification for that specific import from the UI. Leave verification enabled for normal public endpoints.

## Notes

- Data is stored in `./data/dataset_manager.db`.
- The SearxNG runtime config is stored in `./searxng/settings.yml`; replace the development `secret_key` before exposing the service outside your machine.
- Uploaded records are normalized into a canonical internal schema and converted back out on export.
- Export formats intentionally target the common data layouts consumed by Axolotl, Unsloth, and Hugging Face training pipelines.
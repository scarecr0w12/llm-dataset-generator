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
docker compose up --build
```

Open http://localhost:8000.

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

- SearxNG: point the base URL at your instance, for example `http://host.docker.internal:8080`, and optionally fetch full result pages instead of snippets only.
- Web crawl: seed one or more URLs, keep the crawl same-domain by default, and import the extracted page text as draft examples for later annotation.
- GitHub: search repositories, code, or issues via the GitHub REST API. Add a personal access token if you need higher rate limits or private-resource access.

## Notes

- Data is stored in `./data/dataset_manager.db`.
- Uploaded records are normalized into a canonical internal schema and converted back out on export.
- Export formats intentionally target the common data layouts consumed by Axolotl, Unsloth, and Hugging Face training pipelines.
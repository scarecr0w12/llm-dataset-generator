import json
from dataclasses import dataclass
from typing import Any

import httpx


SUPPORTED_LLM_PROVIDERS = {"ollama", "openai", "openai-compatible"}
FINE_TUNE_CAPABLE_PROVIDERS = {"openai", "openai-compatible"}


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model: str = ""
    api_key: str = ""
    organization: str = ""
    project: str = ""
    verify_ssl: bool = True


def validate_provider(provider: str) -> str:
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    return provider


def normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("Base URL is required")
    return normalized


def build_headers(config: LLMConfig) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if config.provider in FINE_TUNE_CAPABLE_PROVIDERS:
        if not config.api_key:
            raise ValueError("API key is required for OpenAI-compatible providers")
        headers["Authorization"] = f"Bearer {config.api_key}"
        if config.organization:
            headers["OpenAI-Organization"] = config.organization
        if config.project:
            headers["OpenAI-Project"] = config.project
    return headers


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
    return ""


def parse_json_payload(text: str) -> Any:
    return json.loads(strip_code_fences(text))


async def post_chat_completion(
    config: LLMConfig,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    async with httpx.AsyncClient(timeout=120.0, verify=config.verify_ssl) as client:
        if config.provider == "ollama":
            response = await client.post(
                f"{normalize_base_url(config.base_url)}/api/chat",
                json={
                    "model": config.model,
                    "stream": False,
                    "messages": messages,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            payload = response.json()
            return extract_text_content(payload.get("message", {}).get("content", ""))

        response = await client.post(
            f"{normalize_base_url(config.base_url)}/v1/chat/completions",
            headers={**build_headers(config), "Content-Type": "application/json"},
            json={
                "model": config.model,
                "temperature": temperature,
                "messages": messages,
            },
        )
        response.raise_for_status()
        payload = response.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return extract_text_content(message.get("content", ""))


async def generate_examples(config: LLMConfig, prompt: str, count: int, temperature: float) -> str:
    system_prompt = (
        "You create high-quality fine-tuning data. "
        "Return strict JSON only. Produce an array of objects with keys: instruction, input, output, labels."
    )
    user_prompt = (
        f"Generate {count} diverse training examples.\n"
        f"Task guidance:\n{prompt}\n"
        "Keep outputs realistic, grounded, and concise."
    )
    return await post_chat_completion(
        config,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )


async def assist_example(
    config: LLMConfig,
    dataset_name: str,
    dataset_description: str,
    example: dict[str, Any],
    action: str,
    instructions: str,
    temperature: float,
) -> dict[str, Any]:
    system_prompt = (
        "You help author fine-tuning datasets. "
        "Return strict JSON only with keys instruction, input, output, system_prompt, labels, status."
    )
    user_prompt = json.dumps(
        {
            "task": "Improve or complete one fine-tuning example",
            "action": action,
            "dataset": {
                "name": dataset_name,
                "description": dataset_description,
            },
            "guidance": instructions,
            "example": {
                "instruction": example.get("instruction", ""),
                "input": example.get("input_text", ""),
                "output": example.get("output_text", ""),
                "system_prompt": example.get("system_prompt", ""),
                "labels": example.get("labels", []),
                "status": example.get("status", "draft"),
            },
            "requirements": [
                "Preserve the example's intent unless the guidance asks for a rewrite.",
                "Use concise, realistic outputs.",
                "Labels must be an array of short strings.",
                "Status must be one of draft, reviewed, approved.",
            ],
        },
        ensure_ascii=True,
    )
    content = await post_chat_completion(
        config,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    data = parse_json_payload(content)
    if not isinstance(data, dict):
        raise ValueError("LLM assist response must be a JSON object")
    return data


async def list_models(config: LLMConfig) -> list[str]:
    async with httpx.AsyncClient(timeout=60.0, verify=config.verify_ssl) as client:
        if config.provider == "ollama":
            response = await client.get(f"{normalize_base_url(config.base_url)}/api/tags")
            response.raise_for_status()
            payload = response.json()
            return sorted(item.get("name", "") for item in payload.get("models", []) if item.get("name"))

        response = await client.get(
            f"{normalize_base_url(config.base_url)}/v1/models",
            headers=build_headers(config),
        )
        response.raise_for_status()
        payload = response.json()
        return sorted(item.get("id", "") for item in payload.get("data", []) if item.get("id"))


async def upload_training_file(
    config: LLMConfig,
    filename: str,
    content: bytes,
    purpose: str = "fine-tune",
) -> dict[str, Any]:
    if config.provider not in FINE_TUNE_CAPABLE_PROVIDERS:
        raise ValueError("Fine-tuning is only available for OpenAI-compatible providers")

    async with httpx.AsyncClient(timeout=120.0, verify=config.verify_ssl) as client:
        response = await client.post(
            f"{normalize_base_url(config.base_url)}/v1/files",
            headers=build_headers(config),
            data={"purpose": purpose},
            files={"file": (filename, content, "application/jsonl")},
        )
        response.raise_for_status()
        return response.json()


async def create_fine_tuning_job(config: LLMConfig, payload: dict[str, Any]) -> dict[str, Any]:
    if config.provider not in FINE_TUNE_CAPABLE_PROVIDERS:
        raise ValueError("Fine-tuning is only available for OpenAI-compatible providers")

    async with httpx.AsyncClient(timeout=120.0, verify=config.verify_ssl) as client:
        response = await client.post(
            f"{normalize_base_url(config.base_url)}/v1/fine_tuning/jobs",
            headers={**build_headers(config), "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def get_fine_tuning_job(config: LLMConfig, job_id: str) -> dict[str, Any]:
    if config.provider not in FINE_TUNE_CAPABLE_PROVIDERS:
        raise ValueError("Fine-tuning is only available for OpenAI-compatible providers")

    async with httpx.AsyncClient(timeout=60.0, verify=config.verify_ssl) as client:
        response = await client.get(
            f"{normalize_base_url(config.base_url)}/v1/fine_tuning/jobs/{job_id}",
            headers=build_headers(config),
        )
        response.raise_for_status()
        return response.json()


async def cancel_fine_tuning_job(config: LLMConfig, job_id: str) -> dict[str, Any]:
    if config.provider not in FINE_TUNE_CAPABLE_PROVIDERS:
        raise ValueError("Fine-tuning is only available for OpenAI-compatible providers")

    async with httpx.AsyncClient(timeout=60.0, verify=config.verify_ssl) as client:
        response = await client.post(
            f"{normalize_base_url(config.base_url)}/v1/fine_tuning/jobs/{job_id}/cancel",
            headers=build_headers(config),
        )
        response.raise_for_status()
        return response.json()

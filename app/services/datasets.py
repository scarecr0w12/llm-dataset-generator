import csv
import hashlib
import io
import json
from typing import Any

import pandas as pd
import tiktoken


SUPPORTED_SCHEMAS = ["alpaca", "sharegpt", "openai"]
SUPPORTED_EXPORTS = ["alpaca", "sharegpt", "openai", "axolotl", "unsloth", "huggingface"]


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def safe_json_loads(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def extract_labels(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    return [item.strip() for item in value.split(",") if item.strip()]


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if record.get("system_prompt"):
        messages.append({"role": "system", "content": record["system_prompt"]})

    user_content = normalize_text(record.get("instruction"))
    input_text = normalize_text(record.get("input_text"))
    if input_text:
        user_content = f"{user_content}\n\nContext:\n{input_text}" if user_content else input_text
    if user_content:
        messages.append({"role": "user", "content": user_content})

    output_text = normalize_text(record.get("output_text"))
    if output_text:
        messages.append({"role": "assistant", "content": output_text})
    return messages


def hash_record(record: dict[str, Any]) -> str:
    parts = [
        normalize_text(record.get("instruction")),
        normalize_text(record.get("input_text")),
        normalize_text(record.get("output_text")),
        normalize_text(record.get("system_prompt")),
        json.dumps(record.get("conversation") or build_messages(record), sort_keys=True),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def estimate_tokens(record: dict[str, Any], tokenizer_name: str = "cl100k_base") -> int:
    text = "\n\n".join(
        part
        for part in [
            normalize_text(record.get("system_prompt")),
            normalize_text(record.get("instruction")),
            normalize_text(record.get("input_text")),
            normalize_text(record.get("output_text")),
        ]
        if part
    )
    if not text:
        return 0

    try:
        encoding = tiktoken.get_encoding(tokenizer_name)
    except ValueError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def canonicalize_record(record: dict[str, Any], tokenizer_name: str = "cl100k_base") -> dict[str, Any]:
    normalized = {
        "instruction": normalize_text(record.get("instruction")),
        "input_text": normalize_text(record.get("input_text")),
        "output_text": normalize_text(record.get("output_text")),
        "system_prompt": normalize_text(record.get("system_prompt")),
        "conversation": record.get("conversation") or [],
        "labels": extract_labels(record.get("labels")),
        "metadata": record.get("metadata") or {},
        "status": normalize_text(record.get("status")) or "draft",
    }
    if not normalized["conversation"]:
        normalized["conversation"] = build_messages(normalized)
    normalized["token_count"] = estimate_tokens(normalized, tokenizer_name=tokenizer_name)
    normalized["content_hash"] = hash_record(normalized)
    return normalized


def derive_instruction_fields(messages: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    system_prompt = ""
    user_messages: list[str] = []
    assistant_messages: list[str] = []

    for message in messages:
        role = message.get("role") or message.get("from")
        content = normalize_text(message.get("content") or message.get("value"))
        if role in {"system"}:
            system_prompt = content
        elif role in {"human", "user"}:
            user_messages.append(content)
        elif role in {"gpt", "assistant"}:
            assistant_messages.append(content)

    instruction = user_messages[-1] if user_messages else ""
    input_text = "\n\n".join(user_messages[:-1]) if len(user_messages) > 1 else ""
    output_text = assistant_messages[-1] if assistant_messages else ""
    return instruction, input_text, output_text, system_prompt


def detect_schema(record: dict[str, Any]) -> str:
    if "messages" in record:
        return "openai"
    if "conversations" in record or "conversation" in record:
        return "sharegpt"
    return "alpaca"


def parse_row(record: dict[str, Any], schema_hint: str | None = None) -> dict[str, Any]:
    schema_name = schema_hint if schema_hint in SUPPORTED_SCHEMAS else detect_schema(record)
    if schema_name == "alpaca":
        return canonicalize_record(
            {
                "instruction": record.get("instruction") or record.get("prompt"),
                "input_text": record.get("input") or record.get("context"),
                "output_text": record.get("output") or record.get("response") or record.get("completion"),
                "system_prompt": record.get("system") or record.get("system_prompt"),
                "metadata": record.get("metadata") or {},
                "labels": record.get("labels") or [],
                "status": record.get("status") or "draft",
            }
        )

    if schema_name == "openai":
        messages = record.get("messages") or []
        instruction, input_text, output_text, system_prompt = derive_instruction_fields(messages)
        return canonicalize_record(
            {
                "instruction": instruction,
                "input_text": input_text,
                "output_text": output_text,
                "system_prompt": system_prompt,
                "conversation": messages,
                "metadata": record.get("metadata") or {},
                "labels": record.get("labels") or [],
                "status": record.get("status") or "draft",
            }
        )

    messages = record.get("conversations") or record.get("conversation") or []
    instruction, input_text, output_text, system_prompt = derive_instruction_fields(messages)
    return canonicalize_record(
        {
            "instruction": instruction,
            "input_text": input_text,
            "output_text": output_text,
            "system_prompt": system_prompt,
            "conversation": messages,
            "metadata": record.get("metadata") or {},
            "labels": record.get("labels") or [],
            "status": record.get("status") or "draft",
        }
    )


def parse_tabular_rows(rows: list[dict[str, Any]], schema_hint: str | None = None) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        cleaned = {str(key): value for key, value in row.items() if value is not None}
        if not cleaned:
            continue
        parsed.append(parse_row(cleaned, schema_hint))
    return parsed


def parse_uploaded_dataset(filename: str, content: bytes, schema_hint: str | None = None) -> list[dict[str, Any]]:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "jsonl":
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        return parse_tabular_rows(rows, schema_hint)
    if suffix == "json":
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("records") or [data]
        return parse_tabular_rows(list(data), schema_hint)
    if suffix == "csv":
        decoded = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))
        return parse_tabular_rows(list(reader), schema_hint)
    if suffix == "parquet":
        frame = pd.read_parquet(io.BytesIO(content))
        return parse_tabular_rows(frame.fillna("").to_dict(orient="records"), schema_hint)
    raise ValueError(f"Unsupported file type: {suffix}")


def export_records(records: list[dict[str, Any]], export_format: str) -> tuple[str, bytes, str]:
    if export_format not in SUPPORTED_EXPORTS:
        raise ValueError(f"Unsupported export format: {export_format}")

    if export_format in {"alpaca", "axolotl", "unsloth"}:
        payload = [
            {
                "instruction": record["instruction"],
                "input": record["input_text"],
                "output": record["output_text"],
                "system": record["system_prompt"],
                "metadata": record["metadata"],
                "labels": record["labels"],
            }
            for record in records
        ]
        return "application/json", json.dumps(payload, indent=2).encode("utf-8"), "json"

    if export_format in {"openai", "huggingface"}:
        lines = []
        for record in records:
            item = {
                "messages": record.get("conversation") or build_messages(record),
                "metadata": record["metadata"],
                "labels": record["labels"],
            }
            lines.append(json.dumps(item, ensure_ascii=True))
        return "application/jsonl", "\n".join(lines).encode("utf-8"), "jsonl"

    payload = []
    for record in records:
        conversations = []
        if record["system_prompt"]:
            conversations.append({"from": "system", "value": record["system_prompt"]})
        conversations.extend(
            [
                {"from": "human", "value": record["instruction"]},
                {"from": "gpt", "value": record["output_text"]},
            ]
        )
        payload.append(
            {
                "conversations": conversations,
                "metadata": record["metadata"],
                "labels": record["labels"],
            }
        )
    return "application/json", json.dumps(payload, indent=2).encode("utf-8"), "json"


def parse_generated_payload(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    data = json.loads(cleaned)
    if isinstance(data, dict):
        data = data.get("examples") or [data]
    return [parse_row(item, "alpaca") for item in data]

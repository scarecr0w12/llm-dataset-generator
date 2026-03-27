import json

import httpx


async def generate_examples(
    provider: str,
    base_url: str,
    model: str,
    prompt: str,
    count: int,
    temperature: float,
) -> str:
    system_prompt = (
        "You create high-quality fine-tuning data. "
        "Return strict JSON only. Produce an array of objects with keys: instruction, input, output, labels."
    )
    user_prompt = (
        f"Generate {count} diverse training examples.\n"
        f"Task guidance:\n{prompt}\n"
        "Keep outputs realistic and concise."
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        if provider == "ollama":
            response = await client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("message", {}).get("content", "[]")

        response = await client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content
        return json.dumps(parsed)

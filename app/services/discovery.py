from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from app.services.datasets import canonicalize_record, normalize_text


DEFAULT_USER_AGENT = "ForgeTune/1.0"
SUPPORTED_GITHUB_SEARCH_TYPES = ["repositories", "code", "issues"]


@dataclass
class SourceDocument:
    title: str
    url: str
    content: str
    snippet: str = ""
    source_type: str = "web"
    metadata: dict[str, Any] = field(default_factory=dict)


def clip_text(value: str, max_chars: int) -> str:
    cleaned = normalize_text(value)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 16].rstrip() + "\n\n[Truncated]"


def split_patterns(value: str) -> list[str]:
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def build_examples_from_documents(
    documents: list[SourceDocument],
    instruction: str,
    system_prompt: str,
    labels: list[str],
    status: str,
    tokenizer_name: str,
) -> list[dict[str, Any]]:
    base_labels = [label for label in labels if normalize_text(label)]
    examples: list[dict[str, Any]] = []

    for document in documents:
        input_parts = []
        if document.title:
            input_parts.append(f"Title: {document.title}")
        if document.url:
            input_parts.append(f"URL: {document.url}")
        if document.snippet and document.snippet not in document.content:
            input_parts.append(f"Snippet: {document.snippet}")
        if document.content:
            input_parts.append(document.content)

        metadata = {
            "source_type": document.source_type,
            "source_url": document.url,
            "source_title": document.title,
            **document.metadata,
        }
        examples.append(
            canonicalize_record(
                {
                    "instruction": instruction,
                    "input_text": "\n\n".join(part for part in input_parts if part),
                    "output_text": "",
                    "system_prompt": system_prompt,
                    "metadata": metadata,
                    "labels": [*base_labels, document.source_type, "external-import"],
                    "status": status,
                },
                tokenizer_name=tokenizer_name,
            )
        )
    return examples


def clean_text_block(value: str) -> str:
    lines = [line.strip() for line in value.splitlines()]
    collapsed: list[str] = []
    blank_pending = False
    for line in lines:
        if not line:
            blank_pending = True
            continue
        if blank_pending and collapsed:
            collapsed.append("")
        collapsed.append(line)
        blank_pending = False
    return "\n".join(collapsed).strip()


def normalize_http_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", parsed.query, ""))


def same_domain(first_url: str, second_url: str) -> bool:
    first_host = urlparse(first_url).netloc.lower()
    second_host = urlparse(second_url).netloc.lower()
    return second_host == first_host or second_host.endswith(f".{first_host}")


def derive_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return slug or parsed.netloc


async def fetch_document(client: httpx.AsyncClient, url: str, max_chars: int) -> tuple[SourceDocument | None, list[str]]:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError:
        return None, []

    resolved_url = str(response.url)
    content_type = response.headers.get("content-type", "").lower()
    body = response.text

    if "text/html" in content_type or "<html" in body[:512].lower():
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "iframe", "form"]):
            tag.decompose()

        container = soup.find("main") or soup.find("article") or soup.body or soup
        title = normalize_text((soup.title.string if soup.title and soup.title.string else "") or derive_title_from_url(resolved_url))
        description_tag = soup.find("meta", attrs={"name": "description"})
        snippet = normalize_text(description_tag.get("content") if description_tag else "")
        content = clip_text(clean_text_block(container.get_text("\n")), max_chars)
        links = [normalize_http_url(urljoin(resolved_url, anchor.get("href", ""))) for anchor in soup.select("a[href]")]
        links = [link for link in links if link]
        return (
            SourceDocument(
                title=title,
                url=resolved_url,
                content=content,
                snippet=clip_text(snippet, min(max_chars, 320)) if snippet else "",
                source_type="web",
                metadata={"content_type": content_type or "text/html"},
            ),
            links,
        )

    if any(token in content_type for token in ["text/", "json", "xml", "javascript"]):
        title = derive_title_from_url(resolved_url)
        content = clip_text(body, max_chars)
        return (
            SourceDocument(
                title=title,
                url=resolved_url,
                content=content,
                snippet=clip_text(content, min(max_chars, 320)),
                source_type="web",
                metadata={"content_type": content_type or "text/plain"},
            ),
            [],
        )

    return None, []


async def import_from_web(
    *,
    urls: list[str],
    max_pages: int,
    max_depth: int,
    same_domain_only: bool,
    include_patterns: str,
    exclude_patterns: str,
    max_chars: int,
    instruction: str,
    system_prompt: str,
    labels: list[str],
    status: str,
    tokenizer_name: str,
) -> list[dict[str, Any]]:
    seeds = [normalize_http_url(url) for url in urls if normalize_http_url(url)]
    if not seeds:
        raise ValueError("At least one valid http(s) URL is required")

    include_filters = split_patterns(include_patterns)
    exclude_filters = split_patterns(exclude_patterns)
    documents: list[SourceDocument] = []

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=30.0,
    ) as client:
        for seed in seeds:
            queue: deque[tuple[str, int]] = deque([(seed, 0)])
            visited: set[str] = set()

            while queue and len(documents) < max_pages:
                current_url, depth = queue.popleft()
                current_url = normalize_http_url(current_url)
                if not current_url or current_url in visited:
                    continue
                visited.add(current_url)

                if exclude_filters and any(pattern in current_url for pattern in exclude_filters):
                    continue
                if include_filters and current_url != seed and not any(pattern in current_url for pattern in include_filters):
                    continue

                document, links = await fetch_document(client, current_url, max_chars)
                if document and document.content:
                    documents.append(document)

                if depth >= max_depth:
                    continue

                for link in links:
                    if same_domain_only and not same_domain(seed, link):
                        continue
                    if link not in visited:
                        queue.append((link, depth + 1))

                if len(documents) >= max_pages:
                    break

    if not documents:
        raise ValueError("No crawlable content was found from the supplied URLs")

    return build_examples_from_documents(
        documents=documents,
        instruction=instruction,
        system_prompt=system_prompt,
        labels=labels,
        status=status,
        tokenizer_name=tokenizer_name,
    )


async def import_from_searxng(
    *,
    base_url: str,
    query: str,
    limit: int,
    categories: str,
    engines: str,
    language: str,
    safesearch: int,
    time_range: str,
    crawl_pages: bool,
    max_chars: int,
    instruction: str,
    system_prompt: str,
    labels: list[str],
    status: str,
    tokenizer_name: str,
) -> list[dict[str, Any]]:
    params = {
        "q": query,
        "format": "json",
        "language": language,
        "safesearch": safesearch,
        "categories": categories,
        "engines": engines,
        "time_range": time_range,
    }
    params = {key: value for key, value in params.items() if value not in {"", None}}

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=30.0,
    ) as client:
        response = await client.get(f"{base_url.rstrip('/')}/search", params=params)
        response.raise_for_status()
        payload = response.json()

        documents: list[SourceDocument] = []
        for result in (payload.get("results") or [])[:limit]:
            target_url = normalize_http_url(result.get("url") or "")
            title = normalize_text(result.get("title")) or derive_title_from_url(target_url)
            snippet = clip_text(result.get("content") or "", min(max_chars, 320))

            if crawl_pages and target_url:
                crawled, _ = await fetch_document(client, target_url, max_chars)
                if crawled and crawled.content:
                    crawled.source_type = "searxng"
                    crawled.metadata.update(
                        {
                            "query": query,
                            "engines": result.get("engines") or [],
                            "category": result.get("category"),
                            "published_date": result.get("publishedDate"),
                        }
                    )
                    documents.append(crawled)
                    continue

            if not target_url and not snippet:
                continue

            documents.append(
                SourceDocument(
                    title=title,
                    url=target_url,
                    content=clip_text("\n\n".join(part for part in [title, snippet, target_url] if part), max_chars),
                    snippet=snippet,
                    source_type="searxng",
                    metadata={
                        "query": query,
                        "engines": result.get("engines") or [],
                        "category": result.get("category"),
                        "published_date": result.get("publishedDate"),
                    },
                )
            )

    if not documents:
        raise ValueError("SearxNG returned no usable results")

    return build_examples_from_documents(
        documents=documents,
        instruction=instruction,
        system_prompt=system_prompt,
        labels=labels,
        status=status,
        tokenizer_name=tokenizer_name,
    )


async def fetch_github_readme(
    client: httpx.AsyncClient,
    base_url: str,
    full_name: str,
    max_chars: int,
) -> str:
    response = await client.get(
        f"{base_url.rstrip('/')}/repos/{full_name}/readme",
        headers={"Accept": "application/vnd.github.raw"},
    )
    if response.status_code >= 400:
        return ""
    return clip_text(response.text, max_chars)


async def import_from_github(
    *,
    base_url: str,
    query: str,
    search_type: str,
    limit: int,
    repository: str,
    sort: str,
    order: str,
    token: str | None,
    include_readme: bool,
    max_chars: int,
    instruction: str,
    system_prompt: str,
    labels: list[str],
    status: str,
    tokenizer_name: str,
) -> list[dict[str, Any]]:
    if search_type not in SUPPORTED_GITHUB_SEARCH_TYPES:
        raise ValueError(f"Unsupported GitHub search type: {search_type}")

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    effective_query = normalize_text(query)
    if repository and search_type in {"code", "issues"} and f"repo:{repository}" not in effective_query:
        effective_query = f"{effective_query} repo:{repository}".strip()

    params = {
        "q": effective_query,
        "per_page": limit,
        "sort": sort,
        "order": order,
    }
    params = {key: value for key, value in params.items() if value not in {"", None}}

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        response = await client.get(f"{base_url.rstrip('/')}/search/{search_type}", params=params)
        response.raise_for_status()
        items = (response.json().get("items") or [])[:limit]
        documents: list[SourceDocument] = []

        for item in items:
            if search_type == "repositories":
                full_name = item.get("full_name") or ""
                description = normalize_text(item.get("description"))
                readme = ""
                if include_readme and full_name:
                    readme = await fetch_github_readme(client, base_url, full_name, max_chars)
                content_parts = [description]
                if item.get("topics"):
                    content_parts.append("Topics: " + ", ".join(item.get("topics") or []))
                if readme:
                    content_parts.append(readme)
                documents.append(
                    SourceDocument(
                        title=full_name,
                        url=item.get("html_url") or "",
                        content=clip_text("\n\n".join(part for part in content_parts if part), max_chars),
                        snippet=description,
                        source_type="github",
                        metadata={
                            "github_type": "repository",
                            "query": effective_query,
                            "stars": item.get("stargazers_count", 0),
                            "language": item.get("language"),
                            "topics": item.get("topics") or [],
                            "default_branch": item.get("default_branch"),
                        },
                    )
                )
                continue

            if search_type == "code":
                raw_response = await client.get(item.get("url") or "", headers={"Accept": "application/vnd.github.raw"})
                if raw_response.status_code >= 400:
                    continue
                repository_payload = item.get("repository") or {}
                full_name = repository_payload.get("full_name") or ""
                path = item.get("path") or ""
                documents.append(
                    SourceDocument(
                        title=f"{full_name}:{path}" if full_name else path,
                        url=item.get("html_url") or repository_payload.get("html_url") or "",
                        content=clip_text(raw_response.text, max_chars),
                        snippet=path,
                        source_type="github",
                        metadata={
                            "github_type": "code",
                            "query": effective_query,
                            "repository": full_name,
                            "path": path,
                            "sha": item.get("sha"),
                        },
                    )
                )
                continue

            documents.append(
                SourceDocument(
                    title=normalize_text(item.get("title")) or "GitHub issue",
                    url=item.get("html_url") or "",
                    content=clip_text(item.get("body") or item.get("title") or "", max_chars),
                    snippet=clip_text(item.get("title") or "", 240),
                    source_type="github",
                    metadata={
                        "github_type": "issue",
                        "query": effective_query,
                        "state": item.get("state"),
                        "comments": item.get("comments", 0),
                        "repository": item.get("repository_url"),
                    },
                )
            )

    if not documents:
        raise ValueError("GitHub returned no usable results")

    return build_examples_from_documents(
        documents=documents,
        instruction=instruction,
        system_prompt=system_prompt,
        labels=labels,
        status=status,
        tokenizer_name=tokenizer_name,
    )
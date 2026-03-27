import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db import engine, get_db
from app.models import Base, Dataset, Example, FineTuneJob, ProviderProfile
from app.services.datasets import (
    SUPPORTED_EXPORTS,
    SUPPORTED_SCHEMAS,
    canonicalize_record,
    export_records,
    parse_generated_payload,
    parse_uploaded_dataset,
    safe_json_loads,
)
from app.services.discovery import SUPPORTED_GITHUB_SEARCH_TYPES, import_from_github, import_from_searxng, import_from_web
from app.services.llm import (
    FINE_TUNE_CAPABLE_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
    LLMConfig,
    assist_example,
    cancel_fine_tuning_job,
    create_fine_tuning_job,
    generate_examples,
    get_fine_tuning_job,
    list_models,
    validate_provider,
    upload_training_file,
)


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(ROOT_DIR / ".env")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DEFAULT_OLLAMA_BASE_URL = os.getenv("FORGETUNE_OLLAMA_BASE_URL", "http://host.docker.internal:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("FORGETUNE_OLLAMA_MODEL", "")
DEFAULT_OPENAI_BASE_URL = os.getenv("FORGETUNE_OPENAI_BASE_URL", "https://api.openai.com")
DEFAULT_OPENAI_API_KEY = os.getenv("FORGETUNE_OPENAI_API_KEY", "")
DEFAULT_OPENAI_ORGANIZATION = os.getenv("FORGETUNE_OPENAI_ORGANIZATION", "")
DEFAULT_OPENAI_PROJECT = os.getenv("FORGETUNE_OPENAI_PROJECT", "")
DEFAULT_OPENAI_MODEL = os.getenv("FORGETUNE_OPENAI_MODEL", "")
DEFAULT_SEARXNG_BASE_URL = os.getenv("FORGETUNE_SEARXNG_BASE_URL", "http://host.docker.internal:8080")
DEFAULT_GITHUB_BASE_URL = os.getenv("FORGETUNE_GITHUB_BASE_URL", "https://api.github.com")
DEFAULT_GITHUB_TOKEN = os.getenv("FORGETUNE_GITHUB_TOKEN", "")
DEFAULT_GITHUB_REPOSITORY = os.getenv("FORGETUNE_GITHUB_REPOSITORY", "")
CRAWLER_SERVICE_URL = os.getenv("FORGETUNE_CRAWLER_SERVICE_URL", "").rstrip("/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="ForgeTune", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    schema_name: str = Field(default="alpaca")


class DatasetUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    schema_name: str = Field(default="alpaca")


class ExamplePayload(BaseModel):
    instruction: str = ""
    input_text: str = ""
    output_text: str = ""
    system_prompt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    labels: list[str] = Field(default_factory=list)
    status: str = "draft"
    conversation: list[dict[str, str]] = Field(default_factory=list)


class ExampleUpdate(ExamplePayload):
    pass


class CurationRequest(BaseModel):
    tokenizer_name: str = "cl100k_base"
    drop_empty: bool = True
    deduplicate: bool = True


class SyntheticRequest(BaseModel):
    provider_profile_id: int | None = None
    provider: str = Field(default="ollama")
    base_url: str = Field(default=DEFAULT_OLLAMA_BASE_URL)
    api_key: str | None = None
    organization: str = DEFAULT_OPENAI_ORGANIZATION
    project: str = DEFAULT_OPENAI_PROJECT
    verify_ssl: bool | None = None
    model: str = ""
    prompt: str
    count: int = Field(default=5, ge=1, le=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ProviderProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    provider_type: str = Field(default="openai-compatible")
    base_url: str = Field(min_length=1)
    default_model: str = ""
    api_key: str | None = None
    organization: str = ""
    project: str = ""
    verify_ssl: bool = True


class ProviderProfileUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    provider_type: str = Field(default="openai-compatible")
    base_url: str = Field(min_length=1)
    default_model: str = ""
    api_key: str | None = None
    organization: str = ""
    project: str = ""
    verify_ssl: bool = True


class AssistRequest(BaseModel):
    provider_profile_id: int
    model: str = ""
    action: str = Field(default="improve-example")
    instructions: str = ""
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)


class FineTuneCreateRequest(BaseModel):
    provider_profile_id: int
    base_model: str = ""
    suffix: str = Field(default="", max_length=40)
    n_epochs: int | None = Field(default=None, ge=1, le=25)


class ExternalImportRequest(BaseModel):
    instruction: str = Field(default="Answer the user using the source context.")
    system_prompt: str = ""
    labels: list[str] = Field(default_factory=list)
    status: str = Field(default="draft")
    tokenizer_name: str = Field(default="cl100k_base")
    max_chars: int = Field(default=6000, ge=500, le=50000)
    verify_ssl: bool = True


class SearxngImportRequest(ExternalImportRequest):
    base_url: str = Field(default=DEFAULT_SEARXNG_BASE_URL)
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=25)
    categories: str = ""
    engines: str = ""
    language: str = Field(default="all")
    safesearch: int = Field(default=1, ge=0, le=2)
    time_range: str = ""
    crawl_pages: bool = False


class WebImportRequest(ExternalImportRequest):
    urls: str = Field(min_length=1)
    max_pages: int = Field(default=5, ge=1, le=50)
    max_depth: int = Field(default=1, ge=0, le=3)
    same_domain_only: bool = True
    include_patterns: str = ""
    exclude_patterns: str = ""


class GitHubImportRequest(ExternalImportRequest):
    base_url: str = Field(default=DEFAULT_GITHUB_BASE_URL)
    query: str = Field(min_length=1)
    search_type: str = Field(default="repositories")
    limit: int = Field(default=5, ge=1, le=25)
    repository: str = DEFAULT_GITHUB_REPOSITORY
    sort: str = ""
    order: str = Field(default="desc")
    token: str | None = DEFAULT_GITHUB_TOKEN or None
    include_readme: bool = True


@dataclass
class DatasetSummary:
    id: int
    name: str
    description: str
    schema_name: str
    example_count: int
    token_total: int
    created_at: str
    updated_at: str


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def mask_secret(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return "*" * len(cleaned)
    return f"{cleaned[:4]}...{cleaned[-4:]}"


def serialize_provider_profile(profile: ProviderProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.name,
        "provider_type": profile.provider_type,
        "base_url": profile.base_url,
        "default_model": profile.default_model,
        "organization": profile.organization,
        "project": profile.project,
        "verify_ssl": profile.verify_ssl,
        "metadata": safe_json_loads(profile.metadata_json, {}),
        "has_api_key": bool(profile.api_key),
        "masked_api_key": mask_secret(profile.api_key),
        "created_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
    }


def serialize_fine_tune_job(job: FineTuneJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "dataset_id": job.dataset_id,
        "provider_profile_id": job.provider_profile_id,
        "provider_name": job.provider_profile.name if job.provider_profile else "",
        "provider_type": job.provider_profile.provider_type if job.provider_profile else "",
        "remote_job_id": job.remote_job_id,
        "remote_file_id": job.remote_file_id,
        "base_model": job.base_model,
        "fine_tuned_model": job.fine_tuned_model,
        "suffix": job.suffix,
        "status": job.status,
        "training_format": job.training_format,
        "training_filename": job.training_filename,
        "hyperparameters": safe_json_loads(job.hyperparameters_json, {}),
        "remote_response": safe_json_loads(job.remote_response_json, {}),
        "error": safe_json_loads(job.error_json, {}),
        "launched_at": job.launched_at.isoformat() if job.launched_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


def serialize_example(example: Example) -> dict[str, Any]:
    return {
        "id": example.id,
        "dataset_id": example.dataset_id,
        "instruction": example.instruction,
        "input_text": example.input_text,
        "output_text": example.output_text,
        "system_prompt": example.system_prompt,
        "conversation": safe_json_loads(example.conversation_json, []),
        "metadata": safe_json_loads(example.metadata_json, {}),
        "labels": safe_json_loads(example.labels_json, []),
        "token_count": example.token_count,
        "status": example.status,
        "content_hash": example.content_hash,
        "created_at": example.created_at.isoformat(),
        "updated_at": example.updated_at.isoformat(),
    }


def provider_profile_or_404(db: Session, provider_profile_id: int) -> ProviderProfile:
    profile = db.get(ProviderProfile, provider_profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Provider profile not found")
    return profile


def fine_tune_job_or_404(db: Session, fine_tune_job_id: int) -> FineTuneJob:
    job = db.get(FineTuneJob, fine_tune_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fine-tune job not found")
    return job


def resolve_runtime_config(
    db: Session,
    provider_profile_id: int | None,
    provider: str,
    base_url: str,
    model: str,
    api_key: str | None,
    organization: str,
    project: str,
    verify_ssl: bool | None,
) -> tuple[LLMConfig, ProviderProfile | None]:
    profile: ProviderProfile | None = None
    if provider_profile_id is not None:
        profile = provider_profile_or_404(db, provider_profile_id)
        provider = profile.provider_type
        base_url = profile.base_url
        api_key = api_key if api_key else profile.api_key
        organization = organization or profile.organization
        project = project or profile.project
        verify_ssl = profile.verify_ssl if verify_ssl is None else verify_ssl
        model = model or profile.default_model

    provider = validate_provider(provider)
    if provider == "openai" and not base_url:
        base_url = DEFAULT_OPENAI_BASE_URL
    if provider == "openai" and not api_key:
        api_key = DEFAULT_OPENAI_API_KEY
    if provider == "openai" and not organization:
        organization = DEFAULT_OPENAI_ORGANIZATION
    if provider == "openai" and not project:
        project = DEFAULT_OPENAI_PROJECT
    if provider == "ollama" and not base_url:
        base_url = DEFAULT_OLLAMA_BASE_URL
    if provider == "openai" and not model:
        model = DEFAULT_OPENAI_MODEL
    if provider == "ollama" and not model:
        model = DEFAULT_OLLAMA_MODEL
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")
    try:
        config = LLMConfig(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=(api_key or "").strip(),
            organization=organization.strip(),
            project=project.strip(),
            verify_ssl=verify_ssl,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return config, profile


def provider_profile_to_config(profile: ProviderProfile, model: str = "") -> LLMConfig:
    api_key = profile.api_key
    if profile.provider_type == "openai" and not api_key:
        api_key = DEFAULT_OPENAI_API_KEY
    organization = profile.organization or (DEFAULT_OPENAI_ORGANIZATION if profile.provider_type == "openai" else "")
    project = profile.project or (DEFAULT_OPENAI_PROJECT if profile.provider_type == "openai" else "")
    resolved_model = model or profile.default_model
    if not resolved_model and profile.provider_type == "openai":
        resolved_model = DEFAULT_OPENAI_MODEL
    if not resolved_model and profile.provider_type == "ollama":
        resolved_model = DEFAULT_OLLAMA_MODEL
    return LLMConfig(
        provider=profile.provider_type,
        base_url=profile.base_url,
        model=resolved_model,
        api_key=api_key,
        organization=organization,
        project=project,
        verify_ssl=profile.verify_ssl,
    )


def update_fine_tune_job_from_remote(job: FineTuneJob, payload: dict[str, Any]) -> None:
    job.status = payload.get("status") or job.status
    job.fine_tuned_model = payload.get("fine_tuned_model") or job.fine_tuned_model
    job.remote_response_json = json.dumps(payload)
    job.error_json = json.dumps(payload.get("error") or {})
    finished_at = payload.get("finished_at")
    if finished_at:
        job.finished_at = datetime.utcfromtimestamp(finished_at)


def summarize_dataset(db: Session, dataset: Dataset) -> DatasetSummary:
    stats = db.execute(
        select(func.count(Example.id), func.coalesce(func.sum(Example.token_count), 0)).where(
            Example.dataset_id == dataset.id
        )
    ).one()
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        schema_name=dataset.schema_name,
        example_count=int(stats[0] or 0),
        token_total=int(stats[1] or 0),
        created_at=dataset.created_at.isoformat(),
        updated_at=dataset.updated_at.isoformat(),
    )


def require_schema(schema_name: str) -> str:
    if schema_name not in SUPPORTED_SCHEMAS:
        raise HTTPException(status_code=400, detail=f"Unsupported schema: {schema_name}")
    return schema_name


def dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def example_or_404(db: Session, example_id: int) -> Example:
    example = db.get(Example, example_id)
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")
    return example


def upsert_example_fields(example: Example, payload: dict[str, Any]) -> None:
    example.instruction = payload["instruction"]
    example.input_text = payload["input_text"]
    example.output_text = payload["output_text"]
    example.system_prompt = payload["system_prompt"]
    example.conversation_json = json.dumps(payload["conversation"])
    example.metadata_json = json.dumps(payload["metadata"])
    example.labels_json = json.dumps(payload["labels"])
    example.token_count = payload["token_count"]
    example.status = payload["status"]
    example.content_hash = payload["content_hash"]


def add_examples(db: Session, dataset_id: int, records: list[dict[str, Any]]) -> int:
    added = 0
    for item in records:
        example = Example(dataset_id=dataset_id)
        upsert_example_fields(example, item)
        db.add(example)
        added += 1
    db.commit()
    return added


async def run_searxng_import(payload: SearxngImportRequest) -> list[dict[str, Any]]:
    return await import_from_searxng(**payload.model_dump())


async def run_web_import(payload: WebImportRequest) -> list[dict[str, Any]]:
    urls = [line.strip() for line in payload.urls.splitlines() if line.strip()]
    return await import_from_web(
        urls=urls,
        max_pages=payload.max_pages,
        max_depth=payload.max_depth,
        same_domain_only=payload.same_domain_only,
        include_patterns=payload.include_patterns,
        exclude_patterns=payload.exclude_patterns,
        max_chars=payload.max_chars,
        instruction=payload.instruction,
        system_prompt=payload.system_prompt,
        labels=payload.labels,
        status=payload.status,
        tokenizer_name=payload.tokenizer_name,
        verify_ssl=payload.verify_ssl,
    )


async def request_crawler_records(path: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not CRAWLER_SERVICE_URL:
        raise ValueError("Crawler service URL is not configured")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            response = await client.post(f"{CRAWLER_SERVICE_URL}{path}", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            detail = exc.response.json().get("detail") or detail
        except ValueError:
            pass
        raise ValueError(detail) from exc
    except httpx.HTTPError as exc:
        raise ValueError(f"Unable to reach crawler service at {CRAWLER_SERVICE_URL}: {exc}") from exc

    records = response.json().get("records")
    if not isinstance(records, list):
        raise ValueError("Crawler service returned an invalid response")
    return records


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "schemas": SUPPORTED_SCHEMAS,
            "exports": SUPPORTED_EXPORTS,
            "github_search_types": SUPPORTED_GITHUB_SEARCH_TYPES,
            "llm_provider_types": sorted(SUPPORTED_LLM_PROVIDERS),
            "default_ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
            "default_ollama_model": DEFAULT_OLLAMA_MODEL,
            "default_openai_base_url": DEFAULT_OPENAI_BASE_URL,
            "default_openai_organization": DEFAULT_OPENAI_ORGANIZATION,
            "default_openai_project": DEFAULT_OPENAI_PROJECT,
            "default_openai_model": DEFAULT_OPENAI_MODEL,
            "default_searxng_base_url": DEFAULT_SEARXNG_BASE_URL,
            "default_github_base_url": DEFAULT_GITHUB_BASE_URL,
            "default_github_repository": DEFAULT_GITHUB_REPOSITORY,
        },
    )


@app.get("/api/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/providers")
async def list_provider_profiles(db: Annotated[Session, Depends(get_db)]) -> list[dict[str, Any]]:
    profiles = db.execute(select(ProviderProfile).order_by(ProviderProfile.updated_at.desc())).scalars().all()
    return [serialize_provider_profile(profile) for profile in profiles]


@app.post("/api/providers")
async def create_provider_profile(
    payload: ProviderProfileCreate,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    provider_type = validate_provider(payload.provider_type)
    profile = ProviderProfile(
        name=payload.name.strip(),
        provider_type=provider_type,
        base_url=payload.base_url.strip(),
        default_model=payload.default_model.strip(),
        api_key=(payload.api_key or "").strip(),
        organization=payload.organization.strip(),
        project=payload.project.strip(),
        verify_ssl=payload.verify_ssl,
    )
    db.add(profile)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create provider profile: {exc}") from exc
    db.refresh(profile)
    return serialize_provider_profile(profile)


@app.get("/api/providers/{provider_profile_id}")
async def get_provider_profile(
    provider_profile_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    profile = provider_profile_or_404(db, provider_profile_id)
    return serialize_provider_profile(profile)


@app.put("/api/providers/{provider_profile_id}")
async def update_provider_profile(
    provider_profile_id: int,
    payload: ProviderProfileUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    profile = provider_profile_or_404(db, provider_profile_id)
    profile.name = payload.name.strip()
    profile.provider_type = validate_provider(payload.provider_type)
    profile.base_url = payload.base_url.strip()
    profile.default_model = payload.default_model.strip()
    if payload.api_key is not None and payload.api_key.strip():
        profile.api_key = payload.api_key.strip()
    profile.organization = payload.organization.strip()
    profile.project = payload.project.strip()
    profile.verify_ssl = payload.verify_ssl
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to update provider profile: {exc}") from exc
    db.refresh(profile)
    return serialize_provider_profile(profile)


@app.delete("/api/providers/{provider_profile_id}")
async def delete_provider_profile(
    provider_profile_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    profile = provider_profile_or_404(db, provider_profile_id)
    if profile.fine_tune_jobs:
        raise HTTPException(status_code=400, detail="Delete linked fine-tune jobs before removing this provider profile")
    db.delete(profile)
    db.commit()
    return {"status": "deleted"}


@app.get("/api/providers/{provider_profile_id}/models")
async def list_provider_models(
    provider_profile_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    profile = provider_profile_or_404(db, provider_profile_id)
    try:
        models = await list_models(provider_profile_to_config(profile))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model listing failed: {exc}") from exc
    return {"models": models}


@app.post("/internal/acquisition/searxng")
async def internal_searxng_import(payload: SearxngImportRequest) -> dict[str, Any]:
    try:
        records = await run_searxng_import(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"records": records}


@app.post("/internal/acquisition/web")
async def internal_web_import(payload: WebImportRequest) -> dict[str, Any]:
    try:
        records = await run_web_import(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"records": records}


@app.get("/api/datasets")
async def list_datasets(db: Annotated[Session, Depends(get_db)]) -> list[dict[str, Any]]:
    datasets = db.execute(select(Dataset).order_by(Dataset.updated_at.desc())).scalars().all()
    return [asdict(summarize_dataset(db, dataset)) for dataset in datasets]


@app.post("/api/datasets")
async def create_dataset(payload: DatasetCreate, db: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    require_schema(payload.schema_name)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Dataset name is required")
    dataset = Dataset(
        name=name,
        description=payload.description.strip(),
        schema_name=payload.schema_name,
    )
    db.add(dataset)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to create dataset: {exc}") from exc
    db.refresh(dataset)
    return asdict(summarize_dataset(db, dataset))


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: int, db: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    dataset = dataset_or_404(db, dataset_id)
    summary = asdict(summarize_dataset(db, dataset))
    summary["metadata"] = safe_json_loads(dataset.metadata_json, {})
    return summary


@app.put("/api/datasets/{dataset_id}")
async def update_dataset(
    dataset_id: int,
    payload: DatasetUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset = dataset_or_404(db, dataset_id)
    require_schema(payload.schema_name)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Dataset name is required")
    dataset.name = name
    dataset.description = payload.description.strip()
    dataset.schema_name = payload.schema_name
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to update dataset: {exc}") from exc
    db.refresh(dataset)
    return asdict(summarize_dataset(db, dataset))


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Annotated[Session, Depends(get_db)]) -> dict[str, str]:
    dataset = dataset_or_404(db, dataset_id)
    db.delete(dataset)
    db.commit()
    return {"status": "deleted"}


@app.get("/api/datasets/{dataset_id}/examples")
async def list_examples(
    dataset_id: int,
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=200, ge=1, le=2000),
) -> list[dict[str, Any]]:
    dataset_or_404(db, dataset_id)
    examples = db.execute(
        select(Example)
        .where(Example.dataset_id == dataset_id)
        .order_by(Example.updated_at.desc())
        .limit(limit)
    ).scalars().all()
    return [serialize_example(example) for example in examples]


@app.post("/api/datasets/{dataset_id}/examples")
async def create_example(
    dataset_id: int,
    payload: ExamplePayload,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset_or_404(db, dataset_id)
    normalized = canonicalize_record(payload.model_dump())
    example = Example(dataset_id=dataset_id)
    upsert_example_fields(example, normalized)
    db.add(example)
    db.commit()
    db.refresh(example)
    return serialize_example(example)


@app.put("/api/examples/{example_id}")
async def update_example(
    example_id: int,
    payload: ExampleUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    example = example_or_404(db, example_id)
    normalized = canonicalize_record(payload.model_dump())
    upsert_example_fields(example, normalized)
    db.commit()
    db.refresh(example)
    return serialize_example(example)


@app.post("/api/examples/{example_id}/assist")
async def assist_with_example(
    example_id: int,
    payload: AssistRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    example = example_or_404(db, example_id)
    dataset = dataset_or_404(db, example.dataset_id)
    try:
        config, profile = resolve_runtime_config(
            db,
            provider_profile_id=payload.provider_profile_id,
            provider="openai-compatible",
            base_url="",
            model=payload.model,
            api_key=None,
            organization="",
            project="",
            verify_ssl=True,
        )
        assisted = await assist_example(
            config=config,
            dataset_name=dataset.name,
            dataset_description=dataset.description,
            example=serialize_example(example),
            action=payload.action,
            instructions=payload.instructions,
            temperature=payload.temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"LLM assist failed: {exc}") from exc

    normalized = canonicalize_record(
        {
            "instruction": assisted.get("instruction") or example.instruction,
            "input_text": assisted.get("input") or assisted.get("input_text") or example.input_text,
            "output_text": assisted.get("output") or assisted.get("output_text") or example.output_text,
            "system_prompt": assisted.get("system_prompt") or example.system_prompt,
            "metadata": {
                **safe_json_loads(example.metadata_json, {}),
                "assisted_by_provider": config.provider,
                "assisted_by_model": config.model,
                "assisted_by_profile": profile.name if profile else None,
                "assist_action": payload.action,
            },
            "labels": assisted.get("labels") or safe_json_loads(example.labels_json, []),
            "status": assisted.get("status") or example.status,
            "conversation": safe_json_loads(example.conversation_json, []),
        }
    )
    upsert_example_fields(example, normalized)
    db.commit()
    db.refresh(example)
    return serialize_example(example)


@app.delete("/api/examples/{example_id}")
async def delete_example(example_id: int, db: Annotated[Session, Depends(get_db)]) -> dict[str, str]:
    example = example_or_404(db, example_id)
    db.delete(example)
    db.commit()
    return {"status": "deleted"}


@app.get("/api/datasets/{dataset_id}/fine-tunes")
async def list_dataset_fine_tunes(
    dataset_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    dataset_or_404(db, dataset_id)
    jobs = db.execute(
        select(FineTuneJob)
        .options(selectinload(FineTuneJob.provider_profile))
        .where(FineTuneJob.dataset_id == dataset_id)
        .order_by(FineTuneJob.updated_at.desc())
    ).scalars().all()
    return [serialize_fine_tune_job(job) for job in jobs]


@app.post("/api/datasets/{dataset_id}/fine-tunes")
async def create_dataset_fine_tune(
    dataset_id: int,
    payload: FineTuneCreateRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset = dataset_or_404(db, dataset_id)
    profile = provider_profile_or_404(db, payload.provider_profile_id)
    if profile.provider_type not in FINE_TUNE_CAPABLE_PROVIDERS:
        raise HTTPException(status_code=400, detail="Selected provider does not support OpenAI-compatible fine-tuning")

    base_model = payload.base_model.strip() or profile.default_model.strip()
    if not base_model:
        raise HTTPException(status_code=400, detail="Base model is required")

    examples = db.execute(
        select(Example)
        .where(Example.dataset_id == dataset_id)
        .order_by(Example.created_at.asc())
    ).scalars().all()
    if not examples:
        raise HTTPException(status_code=400, detail="Dataset has no examples to fine-tune on")

    records = [serialize_example(example) for example in examples]
    _, training_bytes, extension = export_records(records, "openai")
    training_filename = f"{dataset.name.replace(' ', '_').lower()}_train.{extension}"
    config = provider_profile_to_config(profile, base_model)

    request_payload: dict[str, Any] = {
        "training_file": "",
        "model": base_model,
    }
    if payload.suffix.strip():
        request_payload["suffix"] = payload.suffix.strip()
    if payload.n_epochs is not None:
        request_payload["hyperparameters"] = {"n_epochs": payload.n_epochs}

    try:
        uploaded = await upload_training_file(config, training_filename, training_bytes)
        request_payload["training_file"] = uploaded.get("id", "")
        remote_job = await create_fine_tuning_job(config, request_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fine-tune creation failed: {exc}") from exc
    if not request_payload["training_file"]:
        raise HTTPException(status_code=400, detail="Provider did not return a training file ID")
    if not remote_job.get("id"):
        raise HTTPException(status_code=400, detail="Provider did not return a fine-tune job ID")

    job = FineTuneJob(
        dataset_id=dataset_id,
        provider_profile_id=profile.id,
        remote_job_id=remote_job.get("id", ""),
        remote_file_id=uploaded.get("id", ""),
        base_model=base_model,
        fine_tuned_model=remote_job.get("fine_tuned_model") or "",
        suffix=payload.suffix.strip(),
        status=remote_job.get("status") or "queued",
        training_format="openai",
        training_filename=training_filename,
        hyperparameters_json=json.dumps(request_payload.get("hyperparameters") or {}),
        remote_response_json=json.dumps(remote_job),
        error_json=json.dumps(remote_job.get("error") or {}),
        launched_at=utcnow(),
    )
    update_fine_tune_job_from_remote(job, remote_job)
    db.add(job)
    db.commit()
    db.refresh(job)
    db.refresh(profile)
    return serialize_fine_tune_job(job)


@app.post("/api/fine-tunes/{fine_tune_job_id}/sync")
async def sync_fine_tune_job(
    fine_tune_job_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    job = db.execute(
        select(FineTuneJob)
        .options(selectinload(FineTuneJob.provider_profile))
        .where(FineTuneJob.id == fine_tune_job_id)
    ).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Fine-tune job not found")

    profile = job.provider_profile
    config = provider_profile_to_config(profile, job.base_model)
    try:
        remote = await get_fine_tuning_job(config, job.remote_job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fine-tune sync failed: {exc}") from exc
    update_fine_tune_job_from_remote(job, remote)
    db.commit()
    db.refresh(job)
    return serialize_fine_tune_job(job)


@app.post("/api/fine-tunes/{fine_tune_job_id}/cancel")
async def cancel_dataset_fine_tune(
    fine_tune_job_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    job = db.execute(
        select(FineTuneJob)
        .options(selectinload(FineTuneJob.provider_profile))
        .where(FineTuneJob.id == fine_tune_job_id)
    ).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Fine-tune job not found")

    profile = job.provider_profile
    config = provider_profile_to_config(profile, job.base_model)
    try:
        remote = await cancel_fine_tuning_job(config, job.remote_job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fine-tune cancel failed: {exc}") from exc
    update_fine_tune_job_from_remote(job, remote)
    db.commit()
    db.refresh(job)
    return serialize_fine_tune_job(job)


@app.post("/api/datasets/{dataset_id}/import")
async def import_examples(
    dataset_id: int,
    db: Annotated[Session, Depends(get_db)],
    schema_name: Annotated[str, Form()] = "alpaca",
    upload: UploadFile = File(...),
) -> dict[str, Any]:
    dataset = dataset_or_404(db, dataset_id)
    schema_name = require_schema(schema_name)
    raw_content = await upload.read()
    try:
        parsed = parse_uploaded_dataset(upload.filename or "upload", raw_content, schema_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    added = 0
    added = add_examples(db, dataset.id, parsed)
    return {"status": "ok", "imported": added}


@app.post("/api/datasets/{dataset_id}/synthetic")
async def synthetic_examples(
    dataset_id: int,
    payload: SyntheticRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset_or_404(db, dataset_id)
    try:
        config, profile = resolve_runtime_config(
            db,
            provider_profile_id=payload.provider_profile_id,
            provider=payload.provider,
            base_url=payload.base_url,
            model=payload.model,
            api_key=payload.api_key,
            organization=payload.organization,
            project=payload.project,
            verify_ssl=payload.verify_ssl,
        )
        content = await generate_examples(
            config=config,
            prompt=payload.prompt,
            count=payload.count,
            temperature=payload.temperature,
        )
        parsed = parse_generated_payload(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Synthetic generation failed: {exc}") from exc
    for item in parsed:
        item["metadata"] = {
            **(item.get("metadata") or {}),
            "generation_provider": config.provider,
            "generation_model": config.model,
            "provider_profile": profile.name if profile else None,
        }

    added = add_examples(db, dataset_id, parsed)
    return {"status": "ok", "generated": added}


@app.post("/api/datasets/{dataset_id}/sources/searxng")
async def import_searxng_source(
    dataset_id: int,
    payload: SearxngImportRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset_or_404(db, dataset_id)
    try:
        records = (
            await request_crawler_records("/internal/acquisition/searxng", payload.model_dump())
            if CRAWLER_SERVICE_URL
            else await run_searxng_import(payload)
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SearxNG import failed: {exc}") from exc

    imported = add_examples(db, dataset_id, records)
    return {"status": "ok", "source": "searxng", "imported": imported}


@app.post("/api/datasets/{dataset_id}/sources/web")
async def import_web_source(
    dataset_id: int,
    payload: WebImportRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset_or_404(db, dataset_id)
    try:
        records = (
            await request_crawler_records("/internal/acquisition/web", payload.model_dump())
            if CRAWLER_SERVICE_URL
            else await run_web_import(payload)
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Web import failed: {exc}") from exc

    imported = add_examples(db, dataset_id, records)
    return {"status": "ok", "source": "web", "imported": imported}


@app.post("/api/datasets/{dataset_id}/sources/github")
async def import_github_source(
    dataset_id: int,
    payload: GitHubImportRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset_or_404(db, dataset_id)
    if not payload.base_url:
        payload.base_url = DEFAULT_GITHUB_BASE_URL
    if not payload.token and DEFAULT_GITHUB_TOKEN:
        payload.token = DEFAULT_GITHUB_TOKEN
    if not payload.repository and DEFAULT_GITHUB_REPOSITORY:
        payload.repository = DEFAULT_GITHUB_REPOSITORY
    try:
        records = await import_from_github(**payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"GitHub import failed: {exc}") from exc

    imported = add_examples(db, dataset_id, records)
    return {"status": "ok", "source": "github", "imported": imported}


@app.post("/api/datasets/{dataset_id}/curate")
async def curate_dataset(
    dataset_id: int,
    payload: CurationRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    dataset_or_404(db, dataset_id)
    examples = db.execute(select(Example).where(Example.dataset_id == dataset_id)).scalars().all()
    seen_hashes: set[str] = set()
    removed = 0
    updated = 0

    for example in list(examples):
        normalized = canonicalize_record(
            {
                "instruction": example.instruction,
                "input_text": example.input_text,
                "output_text": example.output_text,
                "system_prompt": example.system_prompt,
                "conversation": safe_json_loads(example.conversation_json, []),
                "metadata": safe_json_loads(example.metadata_json, {}),
                "labels": safe_json_loads(example.labels_json, []),
                "status": example.status,
            },
            tokenizer_name=payload.tokenizer_name,
        )
        if payload.drop_empty and not normalized["instruction"] and not normalized["output_text"]:
            db.delete(example)
            removed += 1
            continue
        if payload.deduplicate and normalized["content_hash"] in seen_hashes:
            db.delete(example)
            removed += 1
            continue
        seen_hashes.add(normalized["content_hash"])
        normalized["token_count"] = normalized["token_count"]
        upsert_example_fields(example, normalized)
        updated += 1

    db.commit()
    summary = summarize_dataset(db, dataset_or_404(db, dataset_id))
    return {
        "status": "ok",
        "updated": updated,
        "removed": removed,
        "token_total": summary.token_total,
        "example_count": summary.example_count,
        "tokenizer_name": payload.tokenizer_name,
    }


@app.get("/api/datasets/{dataset_id}/export")
async def export_dataset(
    dataset_id: int,
    export_format: str,
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    dataset = dataset_or_404(db, dataset_id)
    if export_format not in SUPPORTED_EXPORTS:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    examples = db.execute(
        select(Example)
        .options(selectinload(Example.dataset))
        .where(Example.dataset_id == dataset.id)
        .order_by(Example.created_at.asc())
    ).scalars().all()
    records = [serialize_example(example) for example in examples]
    content_type, payload, extension = export_records(records, export_format)
    filename = f"{dataset.name.replace(' ', '_').lower()}_{export_format}.{extension}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=payload, media_type=content_type, headers=headers)


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "favicon.svg", media_type="image/svg+xml")

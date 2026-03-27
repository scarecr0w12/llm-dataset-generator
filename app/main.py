import json
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db import engine, get_db
from app.models import Base, Dataset, Example
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
from app.services.llm import generate_examples


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


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
    provider: str = Field(default="ollama")
    base_url: str = Field(default="http://host.docker.internal:11434")
    model: str
    prompt: str
    count: int = Field(default=5, ge=1, le=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ExternalImportRequest(BaseModel):
    instruction: str = Field(default="Answer the user using the source context.")
    system_prompt: str = ""
    labels: list[str] = Field(default_factory=list)
    status: str = Field(default="draft")
    tokenizer_name: str = Field(default="cl100k_base")
    max_chars: int = Field(default=6000, ge=500, le=50000)


class SearxngImportRequest(ExternalImportRequest):
    base_url: str = Field(default="http://host.docker.internal:8080")
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
    base_url: str = Field(default="https://api.github.com")
    query: str = Field(min_length=1)
    search_type: str = Field(default="repositories")
    limit: int = Field(default=5, ge=1, le=25)
    repository: str = ""
    sort: str = ""
    order: str = Field(default="desc")
    token: str | None = None
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "schemas": SUPPORTED_SCHEMAS,
            "exports": SUPPORTED_EXPORTS,
            "github_search_types": SUPPORTED_GITHUB_SEARCH_TYPES,
        },
    )


@app.get("/api/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


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


@app.delete("/api/examples/{example_id}")
async def delete_example(example_id: int, db: Annotated[Session, Depends(get_db)]) -> dict[str, str]:
    example = example_or_404(db, example_id)
    db.delete(example)
    db.commit()
    return {"status": "deleted"}


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
    if payload.provider not in {"ollama", "openai-compatible"}:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    try:
        content = await generate_examples(
            provider=payload.provider,
            base_url=payload.base_url,
            model=payload.model,
            prompt=payload.prompt,
            count=payload.count,
            temperature=payload.temperature,
        )
        parsed = parse_generated_payload(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Synthetic generation failed: {exc}") from exc

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
        records = await import_from_searxng(**payload.model_dump())
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
    urls = [line.strip() for line in payload.urls.splitlines() if line.strip()]
    try:
        records = await import_from_web(
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

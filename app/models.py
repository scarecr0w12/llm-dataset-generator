from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    schema_name: Mapped[str] = mapped_column(String(50), default="alpaca")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    examples: Mapped[list["Example"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="Example.updated_at.desc()",
    )
    fine_tune_jobs: Mapped[list["FineTuneJob"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="FineTuneJob.updated_at.desc()",
    )


class ProviderProfile(Base):
    __tablename__ = "provider_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    provider_type: Mapped[str] = mapped_column(String(50), nullable=False)
    base_url: Mapped[str] = mapped_column(Text, default="")
    default_model: Mapped[str] = mapped_column(String(255), default="")
    api_key: Mapped[str] = mapped_column(Text, default="")
    organization: Mapped[str] = mapped_column(String(255), default="")
    project: Mapped[str] = mapped_column(String(255), default="")
    verify_ssl: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    fine_tune_jobs: Mapped[list["FineTuneJob"]] = relationship(
        back_populates="provider_profile",
        order_by="FineTuneJob.updated_at.desc()",
    )


class Example(Base):
    __tablename__ = "examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    instruction: Mapped[str] = mapped_column(Text, default="")
    input_text: Mapped[str] = mapped_column(Text, default="")
    output_text: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    conversation_json: Mapped[str] = mapped_column(Text, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    labels_json: Mapped[str] = mapped_column(Text, default="[]")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    dataset: Mapped[Dataset] = relationship(back_populates="examples")


class FineTuneJob(Base):
    __tablename__ = "fine_tune_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    provider_profile_id: Mapped[int] = mapped_column(ForeignKey("provider_profiles.id"), nullable=False, index=True)
    remote_job_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    remote_file_id: Mapped[str] = mapped_column(String(255), default="")
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    fine_tuned_model: Mapped[str] = mapped_column(String(255), default="")
    suffix: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[str] = mapped_column(String(50), default="queued")
    training_format: Mapped[str] = mapped_column(String(50), default="openai")
    training_filename: Mapped[str] = mapped_column(String(255), default="")
    hyperparameters_json: Mapped[str] = mapped_column(Text, default="{}")
    remote_response_json: Mapped[str] = mapped_column(Text, default="{}")
    error_json: Mapped[str] = mapped_column(Text, default="{}")
    launched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    dataset: Mapped[Dataset] = relationship(back_populates="fine_tune_jobs")
    provider_profile: Mapped[ProviderProfile] = relationship(back_populates="fine_tune_jobs")

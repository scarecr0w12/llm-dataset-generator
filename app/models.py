from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
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

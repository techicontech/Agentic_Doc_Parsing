"""Shared typed models for parsed pages / elements."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Route(str, Enum):
    DOCLING = "docling"
    MISTRAL = "mistral"
    SKIP = "skip"


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    PLATE = "plate"
    LIST = "list"
    CAPTION = "caption"
    OTHER = "other"


class ParsedElement(BaseModel):
    type: ElementType
    page: int
    text: str | None = None
    bbox: list[float] | None = None
    table_json: dict[str, Any] | list[Any] | None = None
    figure_id: str | None = None
    caption: str | None = None
    confidence: float | None = None
    extractor_name: str
    extractor_version: str
    # Panel-level figure fields (spec §6.2); None for text elements.
    panel_index: int | None = None
    drawing_code: str | None = None
    linked_step_number: int | None = None
    # Cropped panel image, uploaded to MinIO at persist time.
    image_png: bytes | None = None


class ParsedPage(BaseModel):
    page: int  # 1-based
    route: Route
    elements: list[ParsedElement] = Field(default_factory=list)
    page_image_png: bytes | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class SectionDraft(BaseModel):
    path: list[str]
    title: str
    page_start: int | None = None
    page_end: int | None = None
    doc_code: str | None = None
    edition: str | None = None
    section_kind: str = "other"
    parent_path: list[str] | None = None
    procedure_no: str | None = None
    plate_no: str | None = None
    component_title: str | None = None
    action_title: str | None = None
    citation_key: str | None = None


class PageClassification(BaseModel):
    page: int  # 1-based
    route: Route
    char_count: int = 0
    image_count: int = 0
    drawing_count: int = 0
    reason: str = ""

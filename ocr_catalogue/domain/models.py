from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from typing import Any


@dataclass(frozen=True)
class BBox:
    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2

    def contains_point(self, x: float, y: float, margin: float = 0) -> bool:
        return self.x0 - margin <= x <= self.x1 + margin and self.top - margin <= y <= self.bottom + margin

    def contains(self, other: "BBox", margin: float = 0) -> bool:
        return self.x0 - margin <= other.x0 and self.top - margin <= other.top and self.x1 + margin >= other.x1 and self.bottom + margin >= other.bottom

    def intersects(self, other: "BBox") -> bool:
        return min(self.x1, other.x1) > max(self.x0, other.x0) and min(self.bottom, other.bottom) > max(self.top, other.top)

    def intersection_area(self, other: "BBox") -> float:
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0)) * max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))

    def distance(self, other: "BBox") -> float:
        dx = max(self.x0 - other.x1, other.x0 - self.x1, 0.0)
        dy = max(self.top - other.bottom, other.top - self.bottom, 0.0)
        return hypot(dx, dy)

    def union(self, other: "BBox") -> "BBox":
        return BBox(min(self.x0, other.x0), min(self.top, other.top), max(self.x1, other.x1), max(self.bottom, other.bottom))

    def clip(self, width: float, height: float) -> "BBox":
        return BBox(max(0, self.x0), max(0, self.top), min(width, self.x1), min(height, self.bottom))

    def as_list(self) -> list[float]:
        return [self.x0, self.top, self.x1, self.bottom]


class SemanticRole(str, Enum):
    RAW_TEXT = "RAW_TEXT"
    PRODUCT_TEXT = "PRODUCT_TEXT"
    BRAND = "BRAND"
    ARABIC_TEXT = "ARABIC_TEXT"
    QUANTITY = "QUANTITY"
    MODEL = "MODEL"
    PROMOTION = "PROMOTION"
    PRICE_BASIS = "PRICE_BASIS"
    TECHNICAL_SPEC = "TECHNICAL_SPEC"
    HEADER_FOOTER = "HEADER_FOOTER"
    IMAGE = "IMAGE"
    CONTAINER = "CONTAINER"
    SEPARATOR = "SEPARATOR"


class NumericRole(str, Enum):
    PRICE_MAIN = "PRICE_MAIN"
    CREDIT_PAYMENT = "CREDIT_PAYMENT"
    CASHBACK = "CASHBACK"
    DISCOUNT = "DISCOUNT"
    MODEL = "MODEL"
    POWER = "POWER"
    CAPACITY = "CAPACITY"
    QUANTITY = "QUANTITY"
    PACK_SIZE = "PACK_SIZE"
    DIMENSION = "DIMENSION"
    DURATION = "DURATION"
    PRICE_BASIS = "PRICE_BASIS"
    TECHNICAL_SPEC = "TECHNICAL_SPEC"
    UNKNOWN_NUMBER = "UNKNOWN_NUMBER"


@dataclass
class VisualObject:
    id: str
    page: int
    raw_type: str
    bbox: BBox
    text: str = ""
    font_size: float = 0.0
    font_name: str = ""
    color: Any = None
    language: str = "unknown"
    semantic_role: SemanticRole = SemanticRole.RAW_TEXT
    semantic_confidence: float = 0.0
    image_id: str = ""
    container_id: str = ""
    source_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NumericFact:
    id: str
    page: int
    text: str
    value: str
    bbox: BBox
    role: NumericRole
    confidence: float
    source_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass
class PageScene:
    number: int
    width: float
    height: float
    objects: list[VisualObject] = field(default_factory=list)
    numeric_facts: list[NumericFact] = field(default_factory=list)
    separators: list[VisualObject] = field(default_factory=list)
    raster_path: str = ""
    style_features: dict[str, Any] = field(default_factory=dict)

    def object_by_id(self) -> dict[str, VisualObject]:
        return {obj.id: obj for obj in self.objects}


@dataclass
class CatalogueStyleProfile:
    body_font_size: float = 0.0
    price_font_size: float = 0.0
    percentage_font_size: float = 0.0
    price_fonts: list[str] = field(default_factory=list)
    product_fonts: list[str] = field(default_factory=list)
    repeated_noise: set[str] = field(default_factory=set)
    alignment_modes_x: list[float] = field(default_factory=list)
    alignment_modes_y: list[float] = field(default_factory=list)
    page_profiles: dict[int, dict[str, Any]] = field(default_factory=dict)
    evidence_count: int = 0


@dataclass
class DocumentScene:
    source: str
    pages: list[PageScene]
    style: CatalogueStyleProfile = field(default_factory=CatalogueStyleProfile)


@dataclass
class OfferCandidate:
    id: str
    page: int
    object_ids: list[str] = field(default_factory=list)
    numeric_ids: list[str] = field(default_factory=list)
    bbox: BBox | None = None
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    assignments: dict[str, float] = field(default_factory=dict)


@dataclass
class Offer:
    id: str
    page: int
    bbox: BBox
    object_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    product_name: str = ""
    arabic_name: str = ""
    brand: str = ""
    model: str = ""
    variant: str = ""
    quantity: str = ""
    main_price: str = ""
    percentage: str = ""
    promotion: str = ""
    cashback: str = ""
    price_basis: str = ""
    credit_payment: str = ""
    technical_specs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    crop_mode: str = "offer_graph"
    safe_bbox: list[float] = field(default_factory=list)
    region_quality: dict[str, Any] = field(default_factory=dict)

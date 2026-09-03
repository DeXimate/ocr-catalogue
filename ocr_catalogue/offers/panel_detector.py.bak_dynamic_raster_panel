from __future__ import annotations

"""Card-first detection for structured catalogue pages.

A structured catalogue page already contains the commercial card geometry in
the PDF. On such pages the card is the source of truth for the crop and for
object ownership. Semantic/OCR geometry is only a fallback for free layouts.

Important invariants:

* only PDF vector CONTAINER objects can become native cards;
* PDF IMAGE objects are never used as card boundaries;
* a structured page is discovered from repeated container geometry;
* a native card must contain exactly one PRICE_MAIN centre;
* semantic-core errors are not allowed to shrink a proven native card;
* one-off larger cards are accepted only when their edges align with the
  repeated page structure.
"""

from dataclasses import dataclass
import math
import statistics

from ..domain import BBox, OfferCandidate, PageScene, SemanticRole, VisualObject


@dataclass(frozen=True)
class NativePanel:
    bbox: BBox
    confidence: float
    source_id: str
    source_type: str = "CONTAINER"
    repetition_score: float = 0.0


def _same_geometry(first: BBox, second: BBox) -> bool:
    if min(first.width, first.height, second.width, second.height) <= 0:
        return False
    width_ratio = min(first.width, second.width) / max(first.width, second.width)
    height_ratio = min(first.height, second.height) / max(first.height, second.height)
    area_ratio = min(first.area, second.area) / max(first.area, second.area)
    return width_ratio >= .90 and height_ratio >= .90 and area_ratio >= .84


def _container_candidates(page: PageScene) -> list[VisualObject]:
    """Return deduplicated, card-sized VECTOR containers only."""
    page_area = max(1.0, page.width * page.height)
    raw = []
    for obj in page.objects:
        if obj.semantic_role != SemanticRole.CONTAINER:
            continue
        fraction = obj.bbox.area / page_area
        aspect = max(
            obj.bbox.width / max(1.0, obj.bbox.height),
            obj.bbox.height / max(1.0, obj.bbox.width),
        )
        if not (.006 <= fraction <= .50):
            continue
        if aspect > 5.5:
            continue
        raw.append(obj)

    deduped: list[VisualObject] = []
    for obj in sorted(raw, key=lambda item: item.bbox.area):
        duplicate = next(
            (
                other
                for other in deduped
                if abs(other.bbox.x0 - obj.bbox.x0) <= 1.8
                and abs(other.bbox.top - obj.bbox.top) <= 1.8
                and abs(other.bbox.x1 - obj.bbox.x1) <= 1.8
                and abs(other.bbox.bottom - obj.bbox.bottom) <= 1.8
            ),
            None,
        )
        if duplicate is None:
            deduped.append(obj)
    return deduped


def _geometry_families(containers: list[VisualObject]) -> list[list[VisualObject]]:
    """Connected components of containers with repeated width/height."""
    remaining = list(containers)
    families: list[list[VisualObject]] = []
    while remaining:
        seed = remaining.pop(0)
        family = [seed]
        changed = True
        while changed:
            changed = False
            for obj in list(remaining):
                if any(_same_geometry(obj.bbox, member.bbox) for member in family):
                    family.append(obj)
                    remaining.remove(obj)
                    changed = True
        families.append(family)
    return families


def _spreads_over_page(family: list[VisualObject]) -> bool:
    if len(family) < 2:
        return False
    median_w = statistics.median(obj.bbox.width for obj in family)
    median_h = statistics.median(obj.bbox.height for obj in family)
    xs = [obj.bbox.cx for obj in family]
    ys = [obj.bbox.cy for obj in family]
    return (
        max(xs) - min(xs) >= median_w * .75
        or max(ys) - min(ys) >= median_h * .75
    )


def _qualifying_families(
    page: PageScene,
    containers: list[VisualObject],
    price_count: int,
) -> list[list[VisualObject]]:
    page_area = max(1.0, page.width * page.height)
    families = []
    for family in _geometry_families(containers):
        coverage = sum(obj.bbox.area for obj in family) / page_area
        regular_grid = len(family) >= 4 and coverage >= .18 and _spreads_over_page(family)
        simple_page = (
            2 <= len(family) <= 3
            and len(containers) <= 4
            and price_count <= len(family)
            and coverage >= .40
            and _spreads_over_page(family)
        )
        if regular_grid or simple_page:
            families.append(family)
    return families


def _near(value: float, references: list[float], tolerance: float) -> bool:
    return any(abs(value - reference) <= tolerance for reference in references)


def _aligned_to_structure(box: BBox, reference: list[VisualObject], tolerance: float) -> bool:
    """Allow a one-off spanning card only when all four edges follow the grid."""
    x_edges = [value for obj in reference for value in (obj.bbox.x0, obj.bbox.x1)]
    y_edges = [value for obj in reference for value in (obj.bbox.top, obj.bbox.bottom)]
    return (
        _near(box.x0, x_edges, tolerance)
        and _near(box.x1, x_edges, tolerance)
        and _near(box.top, y_edges, tolerance)
        and _near(box.bottom, y_edges, tolerance)
    )


def _crosses_header_footer(page: PageScene, box: BBox) -> bool:
    for obj in page.objects:
        if obj.semantic_role != SemanticRole.HEADER_FOOTER:
            continue
        overlap = box.intersection_area(obj.bbox)
        if overlap <= 0:
            continue
        ratio = overlap / max(1.0, min(box.area, obj.bbox.area))
        if ratio >= .25:
            return True
    return False


def _structural_panels(page: PageScene, price_boxes: list[BBox]) -> list[NativePanel]:
    containers = _container_candidates(page)
    families = _qualifying_families(page, containers, len(price_boxes))
    if not families:
        return []

    reference = [obj for family in families for obj in family]
    family_ids = {obj.id for obj in reference}
    base_area = statistics.median(obj.bbox.area for obj in reference)
    base_scale = math.sqrt(max(1.0, base_area))
    tolerance = max(2.5, base_scale * .04)

    panels: list[NativePanel] = []
    for obj in containers:
        box = obj.bbox
        ratio = box.area / max(1.0, base_area)
        if not (.50 <= ratio <= 5.0):
            continue
        if _crosses_header_footer(page, box):
            continue

        contained_prices = [
            price for price in price_boxes
            if box.contains_point(price.cx, price.cy, 1.5)
        ]
        if len(contained_prices) != 1:
            continue

        repeated = obj.id in family_ids
        if not repeated and not _aligned_to_structure(box, reference, tolerance):
            continue

        panels.append(
            NativePanel(
                bbox=box,
                confidence=.995 if repeated else .93,
                source_id=obj.id,
                repetition_score=1.0 if repeated else .55,
            )
        )

    unique: list[NativePanel] = []
    for panel in sorted(panels, key=lambda item: item.bbox.area):
        if any(
            abs(other.bbox.x0 - panel.bbox.x0) <= 1.8
            and abs(other.bbox.top - panel.bbox.top) <= 1.8
            and abs(other.bbox.x1 - panel.bbox.x1) <= 1.8
            and abs(other.bbox.bottom - panel.bbox.bottom) <= 1.8
            for other in unique
        ):
            continue
        unique.append(panel)
    return unique


def detect_native_panels(
    page: PageScene,
    candidates: list[OfferCandidate],
    nuclei: dict[str, BBox],
    main_prices: dict[str, BBox],
) -> dict[str, NativePanel]:
    """Map priced offers to exact vector cards.

    ``nuclei`` is intentionally ignored for PANEL_NATIVE detection. A semantic
    mistake must never shrink a card whose vector geometry and price ownership
    are already proven.
    """
    del nuclei

    panels = _structural_panels(page, list(main_prices.values()))
    if not panels:
        return {}

    result: dict[str, NativePanel] = {}
    for candidate in candidates:
        own_price = main_prices.get(candidate.id)
        if own_price is None:
            continue

        matches = [
            panel for panel in panels
            if panel.bbox.contains_point(own_price.cx, own_price.cy, 1.5)
        ]
        if matches:
            result[candidate.id] = min(matches, key=lambda panel: panel.bbox.area)

    return result

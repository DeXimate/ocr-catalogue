from __future__ import annotations

"""Detect native commercial panels before falling back to inferred regions.

Many InDesign catalogues already expose card/frame geometry in the PDF.
When that geometry contains exactly one commercial offer, it is more reliable
than rebuilding the crop from OCR/semantic coordinates.

This module stays conservative:

* a native panel must contain the complete semantic core;
* it must contain the offer's own main price;
* it must not contain another offer's main-price centre or semantic nucleus;
* it must not absorb a repeated header/footer;
* repeated geometry improves confidence but is never mandatory.

If these conditions are not met, the existing FREE_LAYOUT region solver keeps
control.  No catalogue/page coordinates are hardcoded here.
"""

from dataclasses import dataclass
import math

from ..domain import BBox, OfferCandidate, PageScene, SemanticRole, VisualObject


@dataclass(frozen=True)
class NativePanel:
    bbox: BBox
    confidence: float
    source_id: str
    source_type: str
    repetition_score: float = 0.0


def _same_geometry(first: BBox, second: BBox) -> bool:
    """Return True for panels with approximately repeated dimensions."""
    if first.width <= 0 or first.height <= 0 or second.width <= 0 or second.height <= 0:
        return False
    width_ratio = min(first.width, second.width) / max(first.width, second.width)
    height_ratio = min(first.height, second.height) / max(first.height, second.height)
    area_ratio = min(first.area, second.area) / max(first.area, second.area)
    return width_ratio >= .88 and height_ratio >= .88 and area_ratio >= .82


def _visual_candidates(page: PageScene) -> list[VisualObject]:
    page_area = max(1.0, page.width * page.height)
    raw: list[VisualObject] = []
    for obj in page.objects:
        if obj.semantic_role == SemanticRole.CONTAINER:
            if page_area * .002 <= obj.bbox.area <= page_area * .72:
                raw.append(obj)
        elif obj.semantic_role == SemanticRole.IMAGE:
            # An image is only a possible panel background/support.  It is
            # never assumed to be a product photo or a panel by itself.
            fraction = obj.metadata.get("page_fraction", obj.bbox.area / page_area)
            if .006 <= fraction <= .72:
                raw.append(obj)

    # InDesign often emits nearly identical rectangle/image objects. Keep the
    # most structural one (CONTAINER wins over IMAGE) for the same geometry.
    deduped: list[VisualObject] = []
    for obj in sorted(
        raw,
        key=lambda item: (
            0 if item.semantic_role == SemanticRole.CONTAINER else 1,
            item.bbox.area,
        ),
    ):
        duplicate = next(
            (
                other
                for other in deduped
                if abs(other.bbox.x0 - obj.bbox.x0) <= 2
                and abs(other.bbox.top - obj.bbox.top) <= 2
                and abs(other.bbox.x1 - obj.bbox.x1) <= 2
                and abs(other.bbox.bottom - obj.bbox.bottom) <= 2
            ),
            None,
        )
        if duplicate is None:
            deduped.append(obj)
    return deduped


def _repetition_score(panel: VisualObject, candidates: list[VisualObject]) -> float:
    repeated = sum(
        1
        for other in candidates
        if other.id != panel.id and _same_geometry(panel.bbox, other.bbox)
    )
    return min(1.0, repeated / 3.0)


def _crosses_header_footer(page: PageScene, box: BBox) -> bool:
    for obj in page.objects:
        if obj.semantic_role != SemanticRole.HEADER_FOOTER:
            continue
        overlap = box.intersection_area(obj.bbox)
        if overlap <= 0:
            continue
        # A repeated footer/header is often a very thin line. Compare overlap
        # to the smaller object so a panel cannot silently swallow it.
        ratio = overlap / max(1.0, min(box.area, obj.bbox.area))
        if ratio >= .25:
            return True
    return False


def _panel_score(
    page: PageScene,
    panel: VisualObject,
    core: BBox,
    repetition: float,
) -> float:
    page_area = max(1.0, page.width * page.height)
    panel_area = max(1.0, panel.bbox.area)
    core_area = max(1.0, core.area)

    structural = .18 if panel.semantic_role == SemanticRole.CONTAINER else .08
    repeated = .16 * repetition

    # Prefer a panel that is comfortably larger than the semantic nucleus but
    # not a huge page background. This is a score, not a hard card dimension.
    fill_ratio = min(1.0, core_area / panel_area)
    compactness = .16 * min(1.0, fill_ratio * 7.0)
    page_penalty = .18 * max(0.0, panel_area / page_area - .42) / .30

    left_space = max(0.0, core.x0 - panel.bbox.x0)
    right_space = max(0.0, panel.bbox.x1 - core.x1)
    top_space = max(0.0, core.top - panel.bbox.top)
    bottom_space = max(0.0, panel.bbox.bottom - core.bottom)
    margins = [left_space, right_space, top_space, bottom_space]
    margin_balance = min(margins) / max(1.0, max(margins)) if max(margins) else 0.0

    return max(
        0.0,
        min(
            .99,
            .42                         # complete core + own main price
            + structural
            + repeated
            + compactness
            + .07 * margin_balance
            - page_penalty,
        ),
    )


def detect_native_panels(
    page: PageScene,
    candidates: list[OfferCandidate],
    nuclei: dict[str, BBox],
    main_prices: dict[str, BBox],
) -> dict[str, NativePanel]:
    """Return one exclusive native panel for offers where PDF geometry proves it."""
    visual = _visual_candidates(page)
    if not visual:
        return {}

    result: dict[str, NativePanel] = {}
    page_area = max(1.0, page.width * page.height)

    for candidate in candidates:
        core = nuclei.get(candidate.id)
        own_price = main_prices.get(candidate.id)
        if core is None or own_price is None:
            continue

        ranked: list[tuple[float, float, NativePanel]] = []
        for panel in visual:
            box = panel.bbox

            # Native mode is intentionally strict. A panel must enclose the
            # complete semantic nucleus, not only its centre.
            if not box.contains(core, 2.5):
                continue
            if not box.contains_point(own_price.cx, own_price.cy, 2.5):
                continue
            if box.area < max(core.area * 1.03, page_area * .002):
                continue

            foreign_prices = sum(
                box.contains_point(price.cx, price.cy)
                for offer_id, price in main_prices.items()
                if offer_id != candidate.id
            )
            foreign_nuclei = sum(
                box.contains_point(other.cx, other.cy)
                for offer_id, other in nuclei.items()
                if offer_id != candidate.id
            )
            if foreign_prices or foreign_nuclei:
                # This is a shared panel, not an exclusive native offer card.
                # The existing FREE_LAYOUT/shared-support solver will split it.
                continue
            if _crosses_header_footer(page, box):
                continue

            repetition = _repetition_score(panel, visual)
            confidence = _panel_score(page, panel, core, repetition)
            if confidence < .62:
                continue

            native = NativePanel(
                bbox=box,
                confidence=confidence,
                source_id=panel.id,
                source_type=panel.semantic_role.value,
                repetition_score=repetition,
            )
            # Prefer high confidence, then the smallest valid structural panel.
            ranked.append((confidence, -box.area, native))

        if ranked:
            result[candidate.id] = max(ranked, key=lambda item: (item[0], item[1]))[2]

    return result

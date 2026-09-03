from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "ocr_catalogue" / "offers" / "panel_detector.py"
REGION = ROOT / "ocr_catalogue" / "offers" / "region_solver.py"


def backup(path: Path) -> None:
    dest = path.with_suffix(path.suffix + ".bak_dynamic_raster_panel")
    if not dest.exists():
        shutil.copy2(path, dest)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: motif attendu 1 fois, trouvé {count} fois")
    return text.replace(old, new, 1)


PANEL_CONTENT = r'''from __future__ import annotations

"""Dynamic native-card detection for structured catalogue pages.

The important distinction is not VECTOR vs IMAGE. It is whether an object is
proved to be the complete commercial card.

Most cards are vector CONTAINER objects. Some InDesign layouts place a large
featured card as one raster object spanning several normal grid cells. Such a
raster is accepted only when a vector card grid has already been discovered,
all four raster edges align with that grid skeleton, and it contains exactly
one PRICE_MAIN. Ordinary product packshots therefore cannot become boundaries.
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
    page_area = max(1.0, page.width * page.height)
    raw: list[VisualObject] = []
    for obj in page.objects:
        if obj.semantic_role != SemanticRole.CONTAINER:
            continue
        fraction = obj.bbox.area / page_area
        aspect = max(
            obj.bbox.width / max(1.0, obj.bbox.height),
            obj.bbox.height / max(1.0, obj.bbox.width),
        )
        if .006 <= fraction <= .50 and aspect <= 5.5:
            raw.append(obj)

    deduped: list[VisualObject] = []
    for obj in sorted(raw, key=lambda item: item.bbox.area):
        duplicate = next(
            (
                other
                for other in deduped
                if abs(other.bbox.x0 - obj.bbox.x0) <= 1.4
                and abs(other.bbox.top - obj.bbox.top) <= 1.4
                and abs(other.bbox.x1 - obj.bbox.x1) <= 1.4
                and abs(other.bbox.bottom - obj.bbox.bottom) <= 1.4
            ),
            None,
        )
        if duplicate is None:
            deduped.append(obj)
    return deduped


def _geometry_families(containers: list[VisualObject]) -> list[list[VisualObject]]:
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
    families: list[list[VisualObject]] = []
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


def _edge_error(value: float, references: list[float]) -> float:
    return min((abs(value - reference) for reference in references), default=math.inf)


def _aligned_to_structure(box: BBox, reference: list[VisualObject], tolerance: float) -> bool:
    x_edges = [value for obj in reference for value in (obj.bbox.x0, obj.bbox.x1)]
    y_edges = [value for obj in reference for value in (obj.bbox.top, obj.bbox.bottom)]
    return (
        _near(box.x0, x_edges, tolerance)
        and _near(box.x1, x_edges, tolerance)
        and _near(box.top, y_edges, tolerance)
        and _near(box.bottom, y_edges, tolerance)
    )


def _alignment_error(box: BBox, reference: list[VisualObject]) -> float:
    x_edges = [value for obj in reference for value in (obj.bbox.x0, obj.bbox.x1)]
    y_edges = [value for obj in reference for value in (obj.bbox.top, obj.bbox.bottom)]
    return (
        _edge_error(box.x0, x_edges)
        + _edge_error(box.x1, x_edges)
        + _edge_error(box.top, y_edges)
        + _edge_error(box.bottom, y_edges)
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


def _price_count(box: BBox, price_boxes: list[BBox]) -> int:
    return sum(box.contains_point(price.cx, price.cy, 1.5) for price in price_boxes)


def _dedupe_panels(panels: list[NativePanel]) -> list[NativePanel]:
    """Collapse near-identical PDF layers while preserving the outer card."""
    unique: list[NativePanel] = []
    for panel in sorted(panels, key=lambda item: item.bbox.area, reverse=True):
        duplicate = next(
            (
                other
                for other in unique
                if abs(other.bbox.x0 - panel.bbox.x0) <= 3.0
                and abs(other.bbox.top - panel.bbox.top) <= 3.0
                and abs(other.bbox.x1 - panel.bbox.x1) <= 3.0
                and abs(other.bbox.bottom - panel.bbox.bottom) <= 3.0
            ),
            None,
        )
        if duplicate is None:
            unique.append(panel)
    return unique


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

    # Vector cards discovered directly from repeated page geometry.
    for obj in containers:
        box = obj.bbox
        ratio = box.area / max(1.0, base_area)
        if not (.50 <= ratio <= 5.0):
            continue
        if _crosses_header_footer(page, box) or _price_count(box, price_boxes) != 1:
            continue

        repeated = obj.id in family_ids
        if not repeated and not _aligned_to_structure(box, reference, tolerance):
            continue

        panels.append(
            NativePanel(
                bbox=box,
                confidence=.995 if repeated else .94,
                source_id=obj.id,
                source_type="CONTAINER",
                repetition_score=1.0 if repeated else .55,
            )
        )

    # Featured raster cards. IMAGE is admitted only after the vector grid has
    # been proved and all four image edges align with that page's card skeleton.
    page_area = max(1.0, page.width * page.height)
    for image in page.objects:
        if image.semantic_role != SemanticRole.IMAGE:
            continue
        box = image.bbox
        fraction = box.area / page_area
        ratio = box.area / max(1.0, base_area)
        aspect = max(
            box.width / max(1.0, box.height),
            box.height / max(1.0, box.width),
        )
        if not (.006 <= fraction <= .45 and .50 <= ratio <= 6.0 and aspect <= 5.5):
            continue
        if _crosses_header_footer(page, box) or _price_count(box, price_boxes) != 1:
            continue
        if not _aligned_to_structure(box, reference, tolerance):
            continue

        error = _alignment_error(box, reference)
        confidence = max(.90, min(.992, .992 - error / max(1.0, base_scale) * .12))
        panels.append(
            NativePanel(
                bbox=box,
                confidence=confidence,
                source_id=image.id,
                source_type="IMAGE_GRID_PANEL",
                repetition_score=.75,
            )
        )

    return _dedupe_panels(panels)


def detect_native_panels(
    page: PageScene,
    candidates: list[OfferCandidate],
    nuclei: dict[str, BBox],
    main_prices: dict[str, BBox],
) -> dict[str, NativePanel]:
    """Map priced offers to exact cards without trusting semantic-core size."""
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
            panel
            for panel in panels
            if panel.bbox.contains_point(own_price.cx, own_price.cy, 1.5)
        ]
        if not matches:
            continue

        # Highest structural confidence first. Near-identical layers then keep
        # the outermost bbox so a border is not clipped.
        result[candidate.id] = max(
            matches,
            key=lambda panel: (panel.confidence, panel.bbox.area),
        )

    return result
'''


def patch_region() -> None:
    text = REGION.read_text(encoding="utf-8")

    old = '''        shared = image.metadata.get("page_fraction", 0) >= .04
        connected = clipped.intersects(region) or clipped.distance(region) <= math.sqrt(max(1.0, core.area))
        shared_is_partitioned = safe.area <= image.bbox.area * .78
        if (not shared and connected) or (shared and shared_is_partitioned):
            region = region.union(clipped)'''
    new = '''        # A raster is not "shared" just because it occupies a fixed
        # percentage of the page. Large featured cards are legitimate.
        main_prices_inside = sum(
            fact.role == NumericRole.PRICE_MAIN
            and image.bbox.contains_point(fact.bbox.cx, fact.bbox.cy)
            for fact in page.numeric_facts
        )
        shared = main_prices_inside > 1
        connected = clipped.intersects(region) or clipped.distance(region) <= math.sqrt(max(1.0, core.area))
        if connected:
            # The safe region remains the exclusivity fence for truly shared
            # rasters, while single-offer rasters may recover their full visual.
            region = region.union(clipped)'''
    text = replace_once(text, old, new, "replace static shared-image rule")

    old = '''    for image in page.objects:
        if image.semantic_role != SemanticRole.IMAGE or image.metadata.get("page_fraction", 0) < .04:
            continue
        if (image.bbox.contains_point(core.cx, core.cy) or image.bbox.intersects(core)) and safe.area <= image.bbox.area * .78:
            clipped = BBox(max(image.bbox.x0, safe.x0), max(image.bbox.top, safe.top), min(image.bbox.x1, safe.x1), min(image.bbox.bottom, safe.bottom))
            if clipped.width > 0 and clipped.height > 0:
                region = region.union(clipped)'''
    new = '''    for image in page.objects:
        if image.semantic_role != SemanticRole.IMAGE:
            continue
        main_prices_inside = sum(
            fact.role == NumericRole.PRICE_MAIN
            and image.bbox.contains_point(fact.bbox.cx, fact.bbox.cy)
            for fact in page.numeric_facts
        )
        # Only real multi-offer rasters enter this unassigned-support path.
        if main_prices_inside <= 1:
            continue
        if image.bbox.contains_point(core.cx, core.cy) or image.bbox.intersects(core):
            clipped = BBox(max(image.bbox.x0, safe.x0), max(image.bbox.top, safe.top), min(image.bbox.x1, safe.x1), min(image.bbox.bottom, safe.bottom))
            if clipped.width > 0 and clipped.height > 0:
                region = region.union(clipped)'''
    text = replace_once(text, old, new, "replace static unassigned-raster rule")

    old = '''        visual_support = any(
            image.metadata.get("page_fraction", 1) < .04 and image.bbox.area >= core.area * .12
            for image in image_objects
        )'''
    new = '''        visual_support = any(
            image.bbox.area >= core.area * .12
            and sum(
                fact.role == NumericRole.PRICE_MAIN
                and image.bbox.contains_point(fact.bbox.cx, fact.bbox.cy)
                for fact in page.numeric_facts
            ) <= 1
            for image in image_objects
        )'''
    text = replace_once(text, old, new, "replace static visual-support threshold")

    REGION.write_text(text, encoding="utf-8")


def main() -> None:
    for path in (PANEL, REGION):
        if not path.exists():
            raise RuntimeError(f"Fichier introuvable: {path}")
        backup(path)

    PANEL.write_text(PANEL_CONTENT, encoding="utf-8")
    patch_region()

    print("DYNAMIC RASTER PANEL FIX APPLIQUE")
    print(" - ocr_catalogue/offers/panel_detector.py")
    print(" - ocr_catalogue/offers/region_solver.py")
    print()
    print(r"Etape suivante: .\test.ps1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        raise

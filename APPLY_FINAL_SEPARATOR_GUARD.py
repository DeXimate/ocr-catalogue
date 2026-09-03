from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
REGION = ROOT / "ocr_catalogue" / "offers" / "region_solver.py"


def backup(path: Path) -> None:
    dest = path.with_suffix(path.suffix + ".bak_separator_guard")
    if not dest.exists():
        shutil.copy2(path, dest)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: motif attendu 1 fois, trouvé {count} fois")
    return text.replace(old, new, 1)


HELPERS = r'''
def _median(values: list[float], fallback: float = 0.0) -> float:
    values = sorted(values)
    if not values:
        return fallback
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _hard_core(candidate: OfferCandidate, solution: RegionSolution, objects: dict[str, VisualObject], facts: dict[str, NumericFact]) -> BBox:
    # Keep only the offer evidence that must survive a final separator cut.
    # Secondary discount/cashback badges are excluded on purpose because a
    # wrong badge assignment must not prevent a correct physical crop.
    boxes = [
        objects[obj_id].bbox
        for obj_id in candidate.object_ids
        if obj_id in objects
        and objects[obj_id].semantic_role in {
            SemanticRole.PRODUCT_TEXT,
            SemanticRole.BRAND,
            SemanticRole.ARABIC_TEXT,
            SemanticRole.QUANTITY,
            SemanticRole.PRICE_BASIS,
        }
    ]
    boxes += [
        facts[fact_id].bbox
        for fact_id in candidate.numeric_ids
        if fact_id in facts and facts[fact_id].role == NumericRole.PRICE_MAIN
    ]
    return _union(boxes, solution.semantic_core)


def _structural_edge_reference(native_panels: dict[str, object]) -> tuple[list[float], list[float], float, float, float]:
    panels = list(native_panels.values())
    if not panels:
        return [], [], 0.0, 0.0, 0.0
    boxes = [panel.bbox for panel in panels]
    median_width = _median([box.width for box in boxes], 1.0)
    median_height = _median([box.height for box in boxes], 1.0)
    tolerance = max(2.5, min(median_width, median_height) * .04)
    x_edges = [value for box in boxes for value in (box.x0, box.x1)]
    y_edges = [value for box in boxes for value in (box.top, box.bottom)]
    return x_edges, y_edges, median_width, median_height, tolerance


def _near_any(value: float, references: list[float], tolerance: float) -> bool:
    return any(abs(value - reference) <= tolerance for reference in references)


def _removed_strip(region: BBox, coordinate: float, side: str) -> BBox:
    if side == "left":
        return BBox(region.x0, region.top, coordinate, region.bottom)
    if side == "right":
        return BBox(coordinate, region.top, region.x1, region.bottom)
    if side == "top":
        return BBox(region.x0, region.top, region.x1, coordinate)
    return BBox(region.x0, coordinate, region.x1, region.bottom)


def _proposal_after_separator(region: BBox, coordinate: float, side: str) -> BBox:
    if side == "left":
        return BBox(coordinate, region.top, region.x1, region.bottom)
    if side == "right":
        return BBox(region.x0, region.top, coordinate, region.bottom)
    if side == "top":
        return BBox(region.x0, coordinate, region.x1, region.bottom)
    return BBox(region.x0, region.top, region.x1, coordinate)


def _foreign_evidence_in_strip(page: PageScene, candidate: OfferCandidate, strip: BBox, nuclei: dict[str, BBox]) -> bool:
    if any(
        strip.contains_point(core.cx, core.cy)
        for offer_id, core in nuclei.items()
        if offer_id != candidate.id
    ):
        return True

    own_numeric = set(candidate.numeric_ids)
    if any(
        fact.id not in own_numeric
        and fact.role in {NumericRole.PRICE_MAIN, NumericRole.DISCOUNT, NumericRole.CASHBACK}
        and strip.contains_point(fact.bbox.cx, fact.bbox.cy)
        for fact in page.numeric_facts
    ):
        return True

    own_objects = set(candidate.object_ids)
    return any(
        obj.id not in own_objects
        and obj.raw_type == "line"
        and obj.semantic_role in {SemanticRole.PRODUCT_TEXT, SemanticRole.BRAND}
        and strip.contains_point(obj.bbox.cx, obj.bbox.cy)
        for obj in page.objects
    )


def _size_improves_to_native_card(region: BBox, proposal: BBox, side: str, median_width: float, median_height: float) -> bool:
    if side in {"left", "right"}:
        target = median_width
        before = region.width
        after = proposal.width
    else:
        target = median_height
        before = region.height
        after = proposal.height
    if target <= 0:
        return False
    before_error = abs(before - target) / target
    after_error = abs(after - target) / target
    return .62 <= after / target <= 1.42 and after_error + .12 < before_error


def _prune_removed_members(candidate: OfferCandidate, removed: BBox, objects: dict[str, VisualObject], facts: dict[str, NumericFact]) -> None:
    candidate.object_ids = [
        obj_id
        for obj_id in candidate.object_ids
        if obj_id not in objects
        or not removed.contains_point(objects[obj_id].bbox.cx, objects[obj_id].bbox.cy)
        or objects[obj_id].semantic_role == SemanticRole.IMAGE
    ]
    candidate.numeric_ids = [
        fact_id
        for fact_id in candidate.numeric_ids
        if fact_id not in facts
        or facts[fact_id].role == NumericRole.PRICE_MAIN
        or not removed.contains_point(facts[fact_id].bbox.cx, facts[fact_id].bbox.cy)
    ]


def _final_separator_guard(
    page: PageScene,
    candidate: OfferCandidate,
    solution: RegionSolution,
    native_panels: dict[str, object],
    nuclei: dict[str, BBox],
    objects: dict[str, VisualObject],
    facts: dict[str, NumericFact],
) -> None:
    # Structured pages already proved their card geometry somewhere on the
    # page. A non-native crop may therefore snap to repeated structural gutters
    # even when the neighbouring offer was not recognised by the semantic graph.
    if solution.mode == "panel_native" or not native_panels:
        return

    x_edges, y_edges, median_width, median_height, tolerance = _structural_edge_reference(native_panels)
    hard = _hard_core(candidate, solution, objects, facts)
    region = solution.region
    changed = False

    specs = (
        ("left", "vertical", lambda value: region.x0 + tolerance < value < hard.x0 - tolerance, max),
        ("right", "vertical", lambda value: hard.x1 + tolerance < value < region.x1 - tolerance, min),
        ("top", "horizontal", lambda value: region.top + tolerance < value < hard.top - tolerance, max),
        ("bottom", "horizontal", lambda value: hard.bottom + tolerance < value < region.bottom - tolerance, min),
    )

    for side, orientation, between, chooser in specs:
        references = x_edges if orientation == "vertical" else y_edges
        values = [
            separator.bbox.cx if orientation == "vertical" else separator.bbox.cy
            for separator in page.separators
            if separator.metadata.get("orientation") == orientation
        ]
        aligned = [value for value in values if between(value) and _near_any(value, references, tolerance)]
        if not aligned:
            continue

        coordinate = chooser(aligned)
        proposal = _proposal_after_separator(region, coordinate, side)
        if proposal.width <= 0 or proposal.height <= 0 or not proposal.contains(hard, tolerance * .25):
            continue

        removed = _removed_strip(region, coordinate, side)
        foreign_evidence = _foreign_evidence_in_strip(page, candidate, removed, nuclei)
        native_size_evidence = _size_improves_to_native_card(region, proposal, side, median_width, median_height)
        if not (foreign_evidence or native_size_evidence):
            continue

        region = proposal
        _prune_removed_members(candidate, removed, objects, facts)
        solution.boundary_evidence[side] = "final_separator_guard"
        changed = True

    if changed:
        solution.region = region.clip(page.width, page.height)
        solution.mode = "free_layout_guarded"
'''


def patch_region() -> None:
    text = REGION.read_text(encoding="utf-8")

    if "def _final_separator_guard(" not in text:
        anchor = '\ndef _foreign_counts(region: BBox, offer_id: str, nuclei: dict[str, BBox], main_prices: dict[str, BBox]) -> tuple[int, int]:\n'
        text = replace_once(text, anchor, "\n" + HELPERS + anchor, "insert final separator guard")

    if '"final_separator_guard": any(' not in text:
        old = '''        "panel_confidence": round(solution.panel_confidence, 4),\n    }'''
        new = '''        "panel_confidence": round(solution.panel_confidence, 4),\n        "final_separator_guard": any(\n            reason == "final_separator_guard"\n            for reason in solution.boundary_evidence.values()\n        ),\n    }'''
        text = replace_once(text, old, new, "quality guard flag")

    if "# Final invariant for structured pages" not in text:
        old = '''    _resolve_page_conflicts(solutions)\n    _enforce_exclusivity(page, solutions, nuclei, main_prices)\n    _resolve_page_conflicts(solutions)\n    for candidate in candidates:\n        solution = solutions[candidate.id]\n        solution.quality = _quality(page, candidate, solution, nuclei, main_prices, raster)\n'''
        new = '''    _resolve_page_conflicts(solutions)\n    _enforce_exclusivity(page, solutions, nuclei, main_prices)\n    _resolve_page_conflicts(solutions)\n\n    # Final invariant for structured pages: a non-native crop may not keep a\n    # repeated card gutter between its own hard core and an outer crop edge.\n    if native_panels:\n        for candidate in candidates:\n            _final_separator_guard(\n                page, candidate, solutions[candidate.id], native_panels,\n                nuclei, objects, facts,\n            )\n\n    for candidate in candidates:\n        solution = solutions[candidate.id]\n        solution.quality = _quality(page, candidate, solution, nuclei, main_prices, raster)\n'''
        text = replace_once(text, old, new, "call final separator guard")

    REGION.write_text(text, encoding="utf-8")


def main() -> None:
    if not REGION.exists():
        raise RuntimeError(f"Fichier introuvable: {REGION}")
    backup(REGION)
    patch_region()
    print("FINAL SEPARATOR GUARD APPLIQUE")
    print(" - ocr_catalogue/offers/region_solver.py")
    print()
    print(r"Etape suivante: .\test.ps1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        raise

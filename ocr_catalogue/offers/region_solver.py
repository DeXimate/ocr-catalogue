from __future__ import annotations

"""Page-level reconstruction of complete, exclusive commercial offer regions.

The solver deliberately works from page evidence rather than catalogue templates.
Every boundary is inferred from semantic nuclei, PDF geometry and the rendered page.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..domain import BBox, NumericFact, NumericRole, OfferCandidate, PageScene, SemanticRole, VisualObject


CORE_SEMANTIC_ROLES = {
    SemanticRole.PRODUCT_TEXT,
    SemanticRole.BRAND,
    SemanticRole.QUANTITY,
    SemanticRole.PRICE_BASIS,
    SemanticRole.PROMOTION,
    SemanticRole.ARABIC_TEXT,
}
CORE_NUMERIC_ROLES = {
    NumericRole.PRICE_MAIN,
    NumericRole.DISCOUNT,
    NumericRole.PRICE_BASIS,
    NumericRole.CASHBACK,
}
FOREIGN_NUCLEUS_ROLES = {SemanticRole.PRODUCT_TEXT, SemanticRole.BRAND}


@dataclass
class NeighbourMap:
    left: str | None = None
    right: str | None = None
    above: str | None = None
    below: str | None = None


@dataclass
class RegionSolution:
    offer_id: str
    semantic_core: BBox
    safe_region: BBox
    region: BBox
    neighbours: NeighbourMap = field(default_factory=NeighbourMap)
    boundary_evidence: dict[str, str] = field(default_factory=dict)
    quality: dict[str, object] = field(default_factory=dict)


def _union(boxes: Iterable[BBox], fallback: BBox) -> BBox:
    boxes = list(boxes)
    if not boxes:
        return fallback
    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)
    return result


def _semantic_core(candidate: OfferCandidate, objects: dict[str, VisualObject], facts: dict[str, NumericFact]) -> BBox:
    boxes = [
        objects[obj_id].bbox for obj_id in candidate.object_ids
        if obj_id in objects and objects[obj_id].semantic_role in CORE_SEMANTIC_ROLES
    ]
    boxes += [
        facts[fact_id].bbox for fact_id in candidate.numeric_ids
        if fact_id in facts and facts[fact_id].role in CORE_NUMERIC_ROLES
    ]
    fallback = candidate.bbox or BBox(0, 0, 1, 1)
    return _union(boxes, fallback)


def build_offer_nuclei(page: PageScene, candidates: list[OfferCandidate]) -> dict[str, BBox]:
    objects, facts = page.object_by_id(), {fact.id: fact for fact in page.numeric_facts}
    return {candidate.id: _semantic_core(candidate, objects, facts) for candidate in candidates}


def _directional_neighbours(nuclei: dict[str, BBox]) -> dict[str, NeighbourMap]:
    result: dict[str, NeighbourMap] = {}
    for offer_id, core in nuclei.items():
        buckets: dict[str, list[tuple[float, str]]] = {name: [] for name in ("left", "right", "above", "below")}
        for other_id, other in nuclei.items():
            if other_id == offer_id:
                continue
            dx, dy = other.cx - core.cx, other.cy - core.cy
            distance = math.hypot(dx, dy)
            if dx < 0 and abs(dx) >= abs(dy) * .55:
                buckets["left"].append((distance + abs(dy) * .35, other_id))
            if dx > 0 and abs(dx) >= abs(dy) * .55:
                buckets["right"].append((distance + abs(dy) * .35, other_id))
            if dy < 0 and abs(dy) >= abs(dx) * .55:
                buckets["above"].append((distance + abs(dx) * .35, other_id))
            if dy > 0 and abs(dy) >= abs(dx) * .55:
                buckets["below"].append((distance + abs(dx) * .35, other_id))
        result[offer_id] = NeighbourMap(**{
            direction: min(values)[1] if values else None for direction, values in buckets.items()
        })
    return result


class RasterEvidence:
    def __init__(self, page: PageScene):
        self.page = page
        self.gray: np.ndarray | None = None
        if page.raster_path and Path(page.raster_path).exists():
            image = Image.open(page.raster_path).convert("L")
            image.thumbnail((1200, 1800))
            self.gray = np.asarray(image, dtype=np.float32)

    def _pixels(self, box: BBox) -> tuple[int, int, int, int]:
        assert self.gray is not None
        height, width = self.gray.shape
        return (
            max(0, min(width - 1, round(box.x0 / self.page.width * width))),
            max(0, min(height - 1, round(box.top / self.page.height * height))),
            max(1, min(width, round(box.x1 / self.page.width * width))),
            max(1, min(height, round(box.bottom / self.page.height * height))),
        )

    def whitespace_boundary(self, first: BBox, second: BBox, axis: str) -> float | None:
        if self.gray is None:
            return None
        x0, y0, x1, y1 = self._pixels(first.union(second))
        if axis == "x":
            lo = min(first.x1, second.x1)
            hi = max(first.x0, second.x0)
            if hi <= lo:
                lo, hi = sorted((first.cx, second.cx))
            px0 = max(x0, round(lo / self.page.width * self.gray.shape[1]))
            px1 = min(x1, round(hi / self.page.width * self.gray.shape[1]))
            if px1 - px0 < 3:
                return None
            sample = self.gray[y0:y1, px0:px1]
            gradient = np.abs(np.diff(sample, axis=1, prepend=sample[:, :1]))
            activity = gradient.mean(axis=0) + sample.std(axis=0) * .12
            index = self._valley_center(activity)
            return (px0 + index) / self.gray.shape[1] * self.page.width
        lo = min(first.bottom, second.bottom)
        hi = max(first.top, second.top)
        if hi <= lo:
            lo, hi = sorted((first.cy, second.cy))
        py0 = max(y0, round(lo / self.page.height * self.gray.shape[0]))
        py1 = min(y1, round(hi / self.page.height * self.gray.shape[0]))
        if py1 - py0 < 3:
            return None
        sample = self.gray[py0:py1, x0:x1]
        gradient = np.abs(np.diff(sample, axis=0, prepend=sample[:1, :]))
        activity = gradient.mean(axis=1) + sample.std(axis=1) * .12
        index = self._valley_center(activity)
        return (py0 + index) / self.gray.shape[0] * self.page.height

    @staticmethod
    def _valley_center(activity: np.ndarray) -> int:
        if activity.size < 3:
            return activity.size // 2
        smooth_width = max(1, round(activity.size ** .5))
        kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
        smooth = np.convolve(activity, kernel, mode="same")
        threshold = np.percentile(smooth, 30)
        low = smooth <= threshold
        runs: list[tuple[int, int]] = []
        start = None
        for index, value in enumerate(np.append(low, False)):
            if value and start is None: start = index
            elif not value and start is not None:
                runs.append((start, index)); start = None
        if not runs:
            return int(np.argmin(smooth))
        middle = activity.size / 2
        run = min(runs, key=lambda item: (abs((item[0] + item[1]) / 2 - middle), -item[1] + item[0]))
        return round((run[0] + run[1]) / 2)

    def border_activity(self, region: BBox, side: str) -> bool:
        if self.gray is None:
            return False
        x0, y0, x1, y1 = self._pixels(region)
        sample = self.gray[y0:y1, x0:x1]
        if sample.size < 16:
            return False
        width = max(1, round(min(sample.shape) ** .5))
        gradient_x = np.abs(np.diff(sample, axis=1, prepend=sample[:, :1]))
        gradient_y = np.abs(np.diff(sample, axis=0, prepend=sample[:1, :]))
        activity = gradient_x + gradient_y
        border = {"left": activity[:, :width], "right": activity[:, -width:], "top": activity[:width, :], "bottom": activity[-width:, :]}[side]
        return float(np.percentile(border, 75)) > float(np.percentile(activity, 55))


def _separator_between(page: PageScene, first: BBox, second: BBox, axis: str) -> float | None:
    lo, hi = sorted((first.cx, second.cx)) if axis == "x" else sorted((first.cy, second.cy))
    orientation = "vertical" if axis == "x" else "horizontal"
    values = [
        separator.bbox.cx if axis == "x" else separator.bbox.cy
        for separator in page.separators
        if separator.metadata.get("orientation") == orientation
        and lo < (separator.bbox.cx if axis == "x" else separator.bbox.cy) < hi
    ]
    midpoint = (lo + hi) / 2
    return min(values, key=lambda value: abs(value - midpoint), default=None)


def _exclusive_container(page: PageScene, core: BBox, nuclei: dict[str, BBox], offer_id: str) -> BBox | None:
    containers = [
        obj.bbox for obj in page.objects
        if obj.semantic_role == SemanticRole.CONTAINER and obj.bbox.contains(core, 2)
        and obj.bbox.area < page.width * page.height * .85
    ]
    for container in sorted(containers, key=lambda box: box.area):
        foreign = sum(container.contains_point(other.cx, other.cy) for key, other in nuclei.items() if key != offer_id)
        if not foreign:
            return container
    return None


def _boundary(page: PageScene, raster: RasterEvidence, current: BBox, other: BBox, axis: str, container: BBox | None, side: str) -> tuple[float, str]:
    separator = _separator_between(page, current, other, axis)
    if separator is not None:
        return separator, "separateur_visuel"
    if container is not None:
        edge = {"left": container.x0, "right": container.x1, "above": container.top, "below": container.bottom}[side]
        coordinate = other.cx if axis == "x" else other.cy
        own = current.cx if axis == "x" else current.cy
        if min(own, coordinate) <= edge <= max(own, coordinate):
            return edge, "conteneur_exclusif"
    whitespace = raster.whitespace_boundary(current, other, axis)
    if whitespace is not None:
        return whitespace, "vallee_blanche"
    if axis == "x":
        gap = (current.x0 + other.x1) / 2 if side == "left" else (current.x1 + other.x0) / 2
        if (side == "left" and other.x1 <= current.x0) or (side == "right" and current.x1 <= other.x0):
            return gap, "bords_noyaux"
        return (current.cx + other.cx) / 2, "milieu_noyaux"
    gap = (current.top + other.bottom) / 2 if side == "above" else (current.bottom + other.top) / 2
    if (side == "above" and other.bottom <= current.top) or (side == "below" and current.bottom <= other.top):
        return gap, "bords_noyaux"
    return (current.cy + other.cy) / 2, "milieu_noyaux"


def _safe_region(page: PageScene, offer_id: str, core: BBox, nuclei: dict[str, BBox], neighbours: NeighbourMap, raster: RasterEvidence) -> tuple[BBox, dict[str, str]]:
    bounds = {"left": 0.0, "right": page.width, "above": 0.0, "below": page.height}
    evidence = {side: "bord_page" for side in bounds}
    container = _exclusive_container(page, core, nuclei, offer_id)
    if container:
        bounds.update(left=container.x0, right=container.x1, above=container.top, below=container.bottom)
        evidence = {side: "conteneur_exclusif" for side in bounds}
    # Full-page whitespace valleys remain useful when an adjacent offer was
    # not recognized semantically. They are only used on sides without a
    # detected neighbour; between known nuclei `_boundary` already arbitrates
    # separators, containers and raster whitespace in priority order.
    vertical = [separator.bbox for separator in page.separators if separator.metadata.get("orientation") == "vertical"]
    horizontal = [separator.bbox for separator in page.separators if separator.metadata.get("orientation") == "horizontal"]
    if neighbours.left is None:
        candidates_left = [box for box in vertical if box.x1 <= core.x0]
        if candidates_left:
            barrier = max(candidates_left, key=lambda box: box.x1)
            bounds["left"] = max(bounds["left"], barrier.x1); evidence["left"] = "vallee_blanche_page"
    if neighbours.right is None:
        candidates_right = [box for box in vertical if box.x0 >= core.x1]
        if candidates_right:
            barrier = min(candidates_right, key=lambda box: box.x0)
            bounds["right"] = min(bounds["right"], barrier.x0); evidence["right"] = "vallee_blanche_page"
    if neighbours.above is None:
        candidates_above = [box for box in horizontal if box.bottom <= core.top]
        if candidates_above:
            barrier = max(candidates_above, key=lambda box: box.bottom)
            bounds["above"] = max(bounds["above"], barrier.bottom); evidence["above"] = "vallee_blanche_page"
    if neighbours.below is None:
        candidates_below = [box for box in horizontal if box.top >= core.bottom]
        if candidates_below:
            barrier = min(candidates_below, key=lambda box: box.top)
            bounds["below"] = min(bounds["below"], barrier.top); evidence["below"] = "vallee_blanche_page"
    for side, other_id in vars(neighbours).items():
        if not other_id:
            continue
        axis = "x" if side in {"left", "right"} else "y"
        value, reason = _boundary(page, raster, core, nuclei[other_id], axis, container, side)
        if side == "left": bounds[side] = max(bounds[side], min(value, core.x0))
        elif side == "right": bounds[side] = min(bounds[side], max(value, core.x1))
        elif side == "above": bounds[side] = max(bounds[side], min(value, core.top))
        else: bounds[side] = min(bounds[side], max(value, core.bottom))
        evidence[side] = reason
    # Repeated headers and footers are hard page-level barriers.
    for obj in page.objects:
        if obj.semantic_role != SemanticRole.HEADER_FOOTER:
            continue
        overlap = max(0.0, min(core.x1, obj.bbox.x1) - max(core.x0, obj.bbox.x0))
        if overlap / max(1.0, min(core.width, obj.bbox.width)) < .2:
            continue
        if obj.bbox.bottom <= core.top:
            bounds["above"] = max(bounds["above"], obj.bbox.bottom); evidence["above"] = "header_footer"
        elif obj.bbox.top >= core.bottom:
            bounds["below"] = min(bounds["below"], obj.bbox.top); evidence["below"] = "header_footer"
    safe = BBox(bounds["left"], bounds["above"], bounds["right"], bounds["below"]).clip(page.width, page.height)
    return (safe if safe.contains(core) else core.clip(page.width, page.height)), evidence


def _visual_envelope(page: PageScene, candidate: OfferCandidate, core: BBox, safe: BBox, objects: dict[str, VisualObject], raster: RasterEvidence) -> BBox:
    region = core
    assigned = [objects[obj_id] for obj_id in candidate.object_ids if obj_id in objects]
    image_objects = [obj for obj in assigned if obj.semantic_role == SemanticRole.IMAGE]
    # Include local PDF images. A raster shared by several offers is clipped to
    # this offer's safe territory and never imported with its complete bbox.
    for image in image_objects:
        clipped = BBox(max(image.bbox.x0, safe.x0), max(image.bbox.top, safe.top), min(image.bbox.x1, safe.x1), min(image.bbox.bottom, safe.bottom))
        if clipped.width <= 0 or clipped.height <= 0:
            continue
        shared = image.metadata.get("page_fraction", 0) >= .04
        connected = clipped.intersects(region) or clipped.distance(region) <= math.sqrt(max(1.0, core.area))
        shared_is_partitioned = safe.area <= image.bbox.area * .78
        if (not shared and connected) or (shared and shared_is_partitioned):
            region = region.union(clipped)
    # Unassigned large rasters can be visual support for several offers. They
    # are admitted only through the safe-region clip.
    for image in page.objects:
        if image.semantic_role != SemanticRole.IMAGE or image.metadata.get("page_fraction", 0) < .04:
            continue
        if (image.bbox.contains_point(core.cx, core.cy) or image.bbox.intersects(core)) and safe.area <= image.bbox.area * .78:
            clipped = BBox(max(image.bbox.x0, safe.x0), max(image.bbox.top, safe.top), min(image.bbox.x1, safe.x1), min(image.bbox.bottom, safe.bottom))
            if clipped.width > 0 and clipped.height > 0:
                region = region.union(clipped)
    region = BBox(max(region.x0, safe.x0), max(region.top, safe.top), min(region.x1, safe.x1), min(region.bottom, safe.bottom))
    # Iterative completeness: if connected rendered content touches a border,
    # consume the still-unused safe territory in that direction. Convergence is
    # bounded by the four independently inferred safe borders.
    for _ in range(8):
        previous = region
        updates = dict(left=region.x0, right=region.x1, above=region.top, below=region.bottom)
        if region.x0 > safe.x0 and raster.border_activity(region, "left"): updates["left"] = safe.x0
        if region.x1 < safe.x1 and raster.border_activity(region, "right"): updates["right"] = safe.x1
        if region.top > safe.top and raster.border_activity(region, "top"): updates["above"] = safe.top
        if region.bottom < safe.bottom and raster.border_activity(region, "bottom"): updates["below"] = safe.bottom
        region = BBox(updates["left"], updates["above"], updates["right"], updates["below"])
        if region == previous:
            break
    return region


def _foreign_counts(region: BBox, offer_id: str, nuclei: dict[str, BBox], main_prices: dict[str, BBox]) -> tuple[int, int]:
    foreign_nuclei = sum(region.contains_point(core.cx, core.cy) for key, core in nuclei.items() if key != offer_id)
    competing_prices = sum(region.contains_point(box.cx, box.cy) for key, box in main_prices.items() if key != offer_id)
    return foreign_nuclei, competing_prices


def _quality(page: PageScene, candidate: OfferCandidate, solution: RegionSolution, nuclei: dict[str, BBox], main_prices: dict[str, BBox], raster: RasterEvidence) -> dict[str, object]:
    objects, facts = page.object_by_id(), {fact.id: fact for fact in page.numeric_facts}
    semantic_boxes = [objects[obj_id].bbox for obj_id in candidate.object_ids if obj_id in objects and objects[obj_id].semantic_role in CORE_SEMANTIC_ROLES]
    semantic_boxes += [facts[fact_id].bbox for fact_id in candidate.numeric_ids if fact_id in facts and facts[fact_id].role in CORE_NUMERIC_ROLES]
    covered = sum(solution.region.contains(box, .5) for box in semantic_boxes)
    semantic_coverage = covered / max(1, len(semantic_boxes))
    foreign_nuclei, competing_prices = _foreign_counts(solution.region, candidate.id, nuclei, main_prices)
    border_contact = {
        side: raster.border_activity(solution.region, side) and abs(getattr(solution.region, {"left":"x0","right":"x1","top":"top","bottom":"bottom"}[side]) - getattr(solution.safe_region, {"left":"x0","right":"x1","top":"top","bottom":"bottom"}[side])) > 1e-3
        for side in ("left", "right", "top", "bottom")
    }
    visual_completeness = 1.0 - sum(border_contact.values()) / 4
    contamination = min(1.0, foreign_nuclei * .55 + competing_prices * .75)
    accepted = semantic_coverage >= .999 and visual_completeness >= .75 and contamination < .5 and competing_prices == 0
    return {
        "semantic_core": solution.semantic_core.as_list(),
        "safe_region": solution.safe_region.as_list(),
        "boundary_evidence": solution.boundary_evidence,
        "neighbours": vars(solution.neighbours),
        "semantic_coverage": round(semantic_coverage, 4),
        "visual_completeness": round(visual_completeness, 4),
        "foreign_offer_contamination": round(contamination, 4),
        "competing_price_centres": competing_prices,
        "foreign_product_nuclei": foreign_nuclei,
        "border_contact": border_contact,
        "crosses_header_footer": any(obj.semantic_role == SemanticRole.HEADER_FOOTER and solution.region.intersects(obj.bbox) for obj in page.objects),
        "accepted": accepted,
    }


def _main_price_boxes(candidates: list[OfferCandidate], facts: dict[str, NumericFact]) -> dict[str, BBox]:
    result = {}
    for candidate in candidates:
        main = next((facts[item] for item in candidate.numeric_ids if item in facts and facts[item].role == NumericRole.PRICE_MAIN), None)
        if main: result[candidate.id] = main.bbox
    return result


def _resolve_page_conflicts(solutions: dict[str, RegionSolution]) -> None:
    ids = list(solutions)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1:]:
            left, right = solutions[left_id], solutions[right_id]
            if not left.region.intersects(right.region):
                continue
            dx, dy = right.semantic_core.cx - left.semantic_core.cx, right.semantic_core.cy - left.semantic_core.cy
            if abs(dx) >= abs(dy):
                boundary = (left.semantic_core.cx + right.semantic_core.cx) / 2
                first, second = (left, right) if dx > 0 else (right, left)
                proposed_first = BBox(first.region.x0, first.region.top, min(first.region.x1, boundary), first.region.bottom)
                proposed_second = BBox(max(second.region.x0, boundary), second.region.top, second.region.x1, second.region.bottom)
            else:
                boundary = (left.semantic_core.cy + right.semantic_core.cy) / 2
                first, second = (left, right) if dy > 0 else (right, left)
                proposed_first = BBox(first.region.x0, first.region.top, first.region.x1, min(first.region.bottom, boundary))
                proposed_second = BBox(second.region.x0, max(second.region.top, boundary), second.region.x1, second.region.bottom)
            if proposed_first.contains(first.semantic_core): first.region = proposed_first
            if proposed_second.contains(second.semantic_core): second.region = proposed_second


def _clip_foreign_point(region: BBox, core: BBox, x: float, y: float) -> BBox:
    """Clip one foreign centre while preserving the complete semantic core."""
    axes = sorted((("x", abs(x - core.cx)), ("y", abs(y - core.cy))), key=lambda item: item[1], reverse=True)
    for axis, _ in axes:
        if axis == "x" and x > core.x1:
            candidate = BBox(region.x0, region.top, min(region.x1, (core.x1 + x) / 2), region.bottom)
        elif axis == "x" and x < core.x0:
            candidate = BBox(max(region.x0, (core.x0 + x) / 2), region.top, region.x1, region.bottom)
        elif axis == "y" and y > core.bottom:
            candidate = BBox(region.x0, region.top, region.x1, min(region.bottom, (core.bottom + y) / 2))
        elif axis == "y" and y < core.top:
            candidate = BBox(region.x0, max(region.top, (core.top + y) / 2), region.x1, region.bottom)
        else:
            continue
        if candidate.contains(core) and not candidate.contains_point(x, y):
            return candidate
    return region


def _enforce_exclusivity(page: PageScene, solutions: dict[str, RegionSolution], nuclei: dict[str, BBox], main_prices: dict[str, BBox]) -> None:
    """Final invariant pass over all regions after page-level reconciliation."""
    for offer_id, solution in solutions.items():
        region = solution.region
        foreign_points = [
            (box.cx, box.cy) for key, box in main_prices.items()
            if key != offer_id and region.contains_point(box.cx, box.cy)
        ]
        foreign_points += [
            (box.cx, box.cy) for key, box in nuclei.items()
            if key != offer_id and region.contains_point(box.cx, box.cy)
        ]
        for x, y in foreign_points:
            region = _clip_foreign_point(region, solution.semantic_core, x, y)
        for barrier in page.objects:
            if barrier.semantic_role != SemanticRole.HEADER_FOOTER or not region.intersects(barrier.bbox):
                continue
            if barrier.bbox.top >= solution.semantic_core.bottom:
                proposal = BBox(region.x0, region.top, region.x1, min(region.bottom, barrier.bbox.top))
            elif barrier.bbox.bottom <= solution.semantic_core.top:
                proposal = BBox(region.x0, max(region.top, barrier.bbox.bottom), region.x1, region.bottom)
            else:
                continue
            if proposal.contains(solution.semantic_core):
                region = proposal
        solution.region = region.clip(page.width, page.height)


def infer_safe_regions(page: PageScene, candidates: list[OfferCandidate]) -> dict[str, RegionSolution]:
    nuclei = build_offer_nuclei(page, candidates)
    neighbours = _directional_neighbours(nuclei)
    raster = RasterEvidence(page)
    solutions: dict[str, RegionSolution] = {}
    for candidate in candidates:
        core = nuclei[candidate.id]
        safe, boundary_evidence = _safe_region(page, candidate.id, core, nuclei, neighbours[candidate.id], raster)
        solutions[candidate.id] = RegionSolution(candidate.id, core, safe, core, neighbours[candidate.id], boundary_evidence)
    return solutions


def solve_page_regions(page: PageScene, candidates: list[OfferCandidate]) -> dict[str, RegionSolution]:
    objects, facts = page.object_by_id(), {fact.id: fact for fact in page.numeric_facts}
    solutions = infer_safe_regions(page, candidates)
    nuclei = {offer_id: solution.semantic_core for offer_id, solution in solutions.items()}
    raster = RasterEvidence(page)
    for candidate in candidates:
        solution = solutions[candidate.id]
        solution.region = _visual_envelope(page, candidate, solution.semantic_core, solution.safe_region, objects, raster)
    # Page-level second pass: no two independent offers may own the same area.
    _resolve_page_conflicts(solutions)
    main_prices = _main_price_boxes(candidates, facts)
    _enforce_exclusivity(page, solutions, nuclei, main_prices)
    _resolve_page_conflicts(solutions)
    for candidate in candidates:
        solution = solutions[candidate.id]
        solution.quality = _quality(page, candidate, solution, nuclei, main_prices, raster)
    return solutions

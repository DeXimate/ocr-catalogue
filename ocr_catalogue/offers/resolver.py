from __future__ import annotations

import math
import re
import statistics
import uuid
from collections import defaultdict

from ..domain import BBox, DocumentScene, NumericFact, NumericRole, Offer, OfferCandidate, PageScene, SemanticRole, VisualObject
from ..graph import SpatialGraph, build_spatial_graph
from ..semantics import parse_promotion


def _candidate_score(graph: SpatialGraph, seed_id: str, obj_id: str) -> float:
    direct = graph.weight(seed_id, obj_id)
    indirect = max((graph.weight(seed_id, middle) * graph.weight(middle, obj_id) * .78 for middle, _ in graph.neighbours(seed_id, .18)), default=0.0)
    return max(direct, indirect)


def _offer_candidates(page: PageScene, graph: SpatialGraph) -> list[OfferCandidate]:
    main_prices = [fact for fact in page.numeric_facts if fact.role == NumericRole.PRICE_MAIN and fact.confidence >= .5]
    candidates = [OfferCandidate(id=f"offer-{fact.id}", page=page.number, numeric_ids=[fact.id], bbox=fact.bbox, score=fact.confidence, evidence=list(fact.evidence)) for fact in main_prices]
    # A strong product cluster without a plausible price remains visible for
    # review instead of disappearing merely because price OCR failed.
    strong_products = [obj for obj in page.objects if obj.semantic_role == SemanticRole.PRODUCT_TEXT and obj.semantic_confidence >= .64]
    for product in strong_products if not main_prices else []:
        if candidates and max((_candidate_score(graph, fact.id, product.id) for fact in main_prices), default=0) >= .24:
            continue
        candidates.append(OfferCandidate(
            id=f"offer-{product.id}", page=page.number, object_ids=[product.id], bbox=product.bbox,
            score=product.semantic_confidence * .55, evidence=["designation_sans_prix_fiable"], contradictions=["prix_principal_absent"],
        ))
    return candidates


def _assign_objects(page: PageScene, graph: SpatialGraph, candidates: list[OfferCandidate]) -> None:
    facts = {fact.id: fact for fact in page.numeric_facts}
    seeds = {candidate.id: candidate.numeric_ids[0] if candidate.numeric_ids else candidate.object_ids[0] for candidate in candidates}
    eligible = [
        obj for obj in page.objects
        if (obj.raw_type == "line" and obj.semantic_role not in {SemanticRole.RAW_TEXT, SemanticRole.HEADER_FOOTER, SemanticRole.CONTAINER, SemanticRole.SEPARATOR})
    ]
    claims = []
    for obj in eligible:
        for candidate in candidates:
            score = _candidate_score(graph, seeds[candidate.id], obj.id)
            if obj.semantic_role == SemanticRole.BRAND:
                score += .08
            elif obj.semantic_role == SemanticRole.PRODUCT_TEXT:
                score += .06
            elif obj.semantic_role == SemanticRole.IMAGE:
                score -= .04
            claims.append((score, candidate.id, obj.id))
    assigned = set()
    by_candidate = {candidate.id: candidate for candidate in candidates}
    minimums = {SemanticRole.IMAGE: .2, SemanticRole.PRODUCT_TEXT: .2, SemanticRole.BRAND: .18}
    object_map = page.object_by_id()
    for score, candidate_id, object_id in sorted(claims, reverse=True):
        if object_id in assigned:
            continue
        obj = object_map[object_id]
        minimum = minimums.get(obj.semantic_role, .16)
        if score < minimum:
            continue
        by_candidate[candidate_id].object_ids.append(object_id)
        by_candidate[candidate_id].assignments[object_id] = score
        assigned.add(object_id)
    _refine_context_assignments(page, graph, by_candidate, facts, candidates)
    _assign_images(page, by_candidate, facts, candidates)
    # Numeric facts other than main price are also globally exclusive.
    secondary = [fact for fact in page.numeric_facts if fact.role != NumericRole.PRICE_MAIN]
    numeric_claims = []
    for fact in secondary:
        for candidate in candidates:
            seed = seeds[candidate.id]
            score = graph.weight(seed, fact.id)
            if score == 0 and candidate.bbox:
                scale = max(20.0, math.hypot(page.width, page.height) * .08)
                score = math.exp(-candidate.bbox.distance(fact.bbox) / scale) * .5
            numeric_claims.append((score, candidate.id, fact.id))
    assigned_numeric = set()
    for score, candidate_id, fact_id in sorted(numeric_claims, reverse=True):
        if fact_id in assigned_numeric or score < .16:
            continue
        by_candidate[candidate_id].numeric_ids.append(fact_id)
        assigned_numeric.add(fact_id)


def _refine_context_assignments(page: PageScene, graph: SpatialGraph, by_candidate: dict[str, OfferCandidate], facts: dict[str, NumericFact], candidates: list[OfferCandidate]) -> None:
    objects = page.object_by_id()
    body_sizes = [obj.font_size for obj in page.objects if obj.raw_type == "line" and obj.font_size > 0]
    context_radius = max(24.0, statistics.median(body_sizes) * 6.0) if body_sizes else 48.0
    movable = {
        SemanticRole.QUANTITY, SemanticRole.ARABIC_TEXT, SemanticRole.MODEL,
        SemanticRole.PROMOTION, SemanticRole.PRICE_BASIS, SemanticRole.TECHNICAL_SPEC,
    }
    moving = [
        obj_id for candidate in candidates for obj_id in candidate.object_ids
        if obj_id in objects and objects[obj_id].semantic_role in movable
    ]
    for candidate in candidates:
        candidate.object_ids = [obj_id for obj_id in candidate.object_ids if obj_id not in moving]
    for obj_id in moving:
        claims = []
        for candidate in candidates:
            related = [
                current for current in candidate.object_ids
                if current in objects and objects[current].semantic_role in {SemanticRole.PRODUCT_TEXT, SemanticRole.BRAND}
            ]
            main_ids = [fact_id for fact_id in candidate.numeric_ids if fact_id in facts and facts[fact_id].role == NumericRole.PRICE_MAIN]
            relational = max((graph.weight(obj_id, current) for current in related), default=0.0)
            nearest_nucleus = min((objects[obj_id].bbox.distance(objects[current].bbox) for current in related), default=math.inf)
            if nearest_nucleus > context_radius:
                relational = 0.0
            price_link = max((graph.weight(obj_id, fact_id) for fact_id in main_ids), default=0.0)
            # Context must be supported by the product/brand nucleus.  A weak
            # price-only relation is not enough: large headings often sit in
            # the same column as the first offer of a page.
            role = objects[obj_id].semantic_role
            if role in {SemanticRole.QUANTITY, SemanticRole.ARABIC_TEXT, SemanticRole.MODEL, SemanticRole.TECHNICAL_SPEC}:
                score = relational
            else:
                score = max(relational * .9 + price_link * .1, price_link * .72)
            claims.append((score, candidate))
        score, owner = max(claims, key=lambda item: item[0])
        if score >= .18:
            owner.object_ids.append(obj_id)
            owner.assignments[obj_id] = score


def _assign_images(page: PageScene, by_candidate: dict[str, OfferCandidate], facts: dict[str, NumericFact], candidates: list[OfferCandidate]) -> None:
    cores = {}
    semantic_cores = {}
    objects = page.object_by_id()
    for candidate in candidates:
        boxes = [facts[fact_id].bbox for fact_id in candidate.numeric_ids if fact_id in facts and facts[fact_id].role == NumericRole.PRICE_MAIN]
        boxes += [objects[obj_id].bbox for obj_id in candidate.object_ids if obj_id in objects]
        if boxes:
            core = boxes[0]
            for box in boxes[1:]: core = core.union(box)
            cores[candidate.id] = core
        semantic_boxes = [
            objects[obj_id].bbox for obj_id in candidate.object_ids
            if obj_id in objects and objects[obj_id].semantic_role in {SemanticRole.PRODUCT_TEXT, SemanticRole.BRAND, SemanticRole.QUANTITY}
        ]
        if semantic_boxes:
            semantic = semantic_boxes[0]
            for box in semantic_boxes[1:]: semantic = semantic.union(box)
            semantic_cores[candidate.id] = semantic
    images = [obj for obj in page.objects if obj.semantic_role == SemanticRole.IMAGE and .00015 <= obj.metadata.get("page_fraction", 1) <= .18 and max(obj.bbox.width / max(1, obj.bbox.height), obj.bbox.height / max(1, obj.bbox.width)) <= 9]
    claims = []
    scale = max(20.0, math.hypot(page.width, page.height) * .1)
    for image in images:
        for candidate_id, core in cores.items():
            semantic = semantic_cores.get(candidate_id, core)
            overlap_x = max(0.0, min(image.bbox.x1, semantic.x1) - max(image.bbox.x0, semantic.x0))
            overlap_y = max(0.0, min(image.bbox.bottom, semantic.bottom) - max(image.bbox.top, semantic.top))
            if image.bbox.distance(semantic) > scale * .82:
                continue
            if overlap_x <= 0 and not (overlap_y > 0 and image.bbox.distance(semantic) <= scale * .5):
                continue
            if abs(image.bbox.cx - core.cx) > max(core.width * 1.25, image.bbox.width * .42):
                continue
            if image.bbox.width > core.width * 2.7 and image.bbox.height < core.height * 1.4:
                continue
            nearby_offers = sum(image.bbox.contains_point(other.cx, other.cy) for other in cores.values())
            if nearby_offers > 1:
                continue
            proximity = math.exp(-image.bbox.distance(core) / scale)
            align_x = math.exp(-abs(image.bbox.cx - core.cx) / max(12.0, image.bbox.width + core.width))
            score = .68 * proximity + .32 * align_x
            claims.append((score, candidate_id, image.id))
    assigned = set()
    candidate_images: dict[str, list[BBox]] = defaultdict(list)
    for score, candidate_id, image_id in sorted(claims, reverse=True):
        if image_id in assigned or score < .55:
            continue
        image_box = objects[image_id].bbox
        existing = candidate_images[candidate_id]
        if existing:
            connected = False
            for box in existing:
                overlap = box.intersection_area(image_box) / max(1.0, min(box.area, image_box.area))
                centres_aligned = abs(box.cx - image_box.cx) <= max(box.width, image_box.width) * .7
                if overlap >= .035 or (box.distance(image_box) <= scale * .1 and centres_aligned):
                    connected = True
                    break
            if not connected:
                continue
        by_candidate[candidate_id].object_ids.append(image_id)
        by_candidate[candidate_id].assignments[image_id] = score
        candidate_images[candidate_id].append(image_box)
        assigned.add(image_id)


def _brand(text: str) -> str:
    match = re.search(r"[“\"]([^”\"]+)[”\"]", text)
    return match.group(1).strip() if match else ""


def _pick_product(objects: list[VisualObject], price: NumericFact | None, style) -> str:
    candidates = [obj for obj in objects if obj.semantic_role == SemanticRole.PRODUCT_TEXT and re.search(r"[A-Za-zÀ-ÿ]", obj.text)]
    if not candidates:
        return ""
    def score(obj: VisualObject) -> float:
        distance = price.bbox.distance(obj.bbox) if price else 0
        proximity = math.exp(-distance / max(12.0, style.body_font_size * 8))
        font_match = 1.0 if obj.font_name in style.product_fonts else .55
        useful_length = 1.0 - min(1.0, abs(len(obj.text) - 20) / 80)
        noise = .7 if re.search(r"variétés|existe en|parfums|go[uû]ts au choix|photos", obj.text, re.I) else 0
        return .4 * obj.semantic_confidence + .3 * proximity + .15 * font_match + .15 * useful_length - noise
    primary = max(candidates, key=score)
    selected = [primary]
    for obj in candidates:
        if obj.id == primary.id:
            continue
        vertical_gap = max(obj.bbox.top - primary.bbox.bottom, primary.bbox.top - obj.bbox.bottom, 0)
        horizontal_overlap = max(0.0, min(obj.bbox.x1, primary.bbox.x1) - max(obj.bbox.x0, primary.bbox.x0))
        centres_close = abs(obj.bbox.cx - primary.bbox.cx) <= max(primary.bbox.width, obj.bbox.width) * .65
        if vertical_gap <= max(primary.font_size, obj.font_size, style.body_font_size) * 1.5 and (horizontal_overlap > 0 or centres_close):
            selected.append(obj)
    return " ".join(dict.fromkeys(obj.text.strip() for obj in sorted(selected, key=lambda item: (item.bbox.cy, item.bbox.x0))))


def _smallest_container(page: PageScene, key_box: BBox, centers: list[tuple[str, float, float]]) -> tuple[BBox | None, int]:
    vector = [obj.bbox for obj in page.objects if obj.semantic_role == SemanticRole.CONTAINER and obj.bbox.contains(key_box, 3)]
    visual = [
        obj.bbox for obj in page.objects
        if obj.semantic_role == SemanticRole.IMAGE and .05 <= obj.metadata.get("page_fraction", 0) <= .72 and obj.bbox.contains(key_box, 3)
    ]
    containers = visual or vector
    if not containers:
        return None, 0
    container = min(containers, key=lambda bbox: bbox.area)
    count = sum(container.contains_point(x, y) for _, x, y in centers)
    return container, count


def _partition_container(container: BBox, current: tuple[float, float], others: list[tuple[float, float]]) -> BBox:
    x0, top, x1, bottom = container.x0, container.top, container.x1, container.bottom
    cx, cy = current
    local = [(x, y) for x, y in others if container.contains_point(x, y)]
    for ox, oy in local:
        dx, dy = abs(ox - cx), abs(oy - cy)
        if max(dx, dy) < min(dx, dy) * 1.5:
            continue
        if dx >= dy:
            boundary = (cx + ox) / 2
            if ox < cx:
                x0 = max(x0, boundary)
            else:
                x1 = min(x1, boundary)
        else:
            boundary = (cy + oy) / 2
            if oy < cy:
                top = max(top, boundary)
            else:
                bottom = min(bottom, boundary)
    return BBox(x0, top, x1, bottom)


def _snap_to_separators(bbox: BBox, page: PageScene) -> BBox:
    x0, top, x1, bottom = bbox.x0, bbox.top, bbox.x1, bbox.bottom
    tolerance_x = max(4.0, bbox.width * .12)
    tolerance_y = max(4.0, bbox.height * .12)
    vertical = [separator.bbox.cx for separator in page.separators if separator.metadata.get("orientation") == "vertical"]
    horizontal = [separator.bbox.cy for separator in page.separators if separator.metadata.get("orientation") == "horizontal"]
    for current, setter in ((x0, "x0"), (x1, "x1")):
        nearest = min(vertical, key=lambda value: abs(value - current), default=None)
        if nearest is not None and abs(nearest - current) <= tolerance_x:
            if setter == "x0": x0 = nearest
            else: x1 = nearest
    for current, setter in ((top, "top"), (bottom, "bottom")):
        nearest = min(horizontal, key=lambda value: abs(value - current), default=None)
        if nearest is not None and abs(nearest - current) <= tolerance_y:
            if setter == "top": top = nearest
            else: bottom = nearest
    return BBox(x0, top, x1, bottom)


def _clip_to_header_footer(region: BBox, essential_core: BBox, page: PageScene) -> BBox:
    """Prevent an offer crop from crossing a semantic header/footer band."""
    result = region
    margin = max(1.5, min(page.width, page.height) * .003)
    barriers = [
        obj for obj in page.objects
        if obj.semantic_role == SemanticRole.HEADER_FOOTER and obj.raw_type == "line"
    ]
    for barrier in barriers:
        box = barrier.bbox
        overlap_x = max(0.0, min(result.x1, box.x1) - max(result.x0, box.x0))
        if overlap_x / max(1.0, min(result.width, box.width)) < .25:
            continue
        if box.top >= essential_core.bottom and result.bottom > box.top:
            candidate = BBox(result.x0, result.top, result.x1, max(essential_core.bottom, box.top - margin))
            if candidate.contains(essential_core):
                result = candidate
        elif box.bottom <= essential_core.top and result.top < box.bottom:
            candidate = BBox(result.x0, min(essential_core.top, box.bottom + margin), result.x1, result.bottom)
            if candidate.contains(essential_core):
                result = candidate
    return result.clip(page.width, page.height)


def _partition_visual_support(
    support: BBox,
    origin: BBox,
    essential_core: BBox,
    own_main: NumericFact | None,
    page: PageScene,
    all_cores: dict[str, BBox],
    candidate_id: str,
) -> tuple[BBox, bool]:
    """Return the local part of a raster shared by neighbouring offers."""
    current_x = own_main.bbox.cx if own_main else origin.cx
    current_y = own_main.bbox.cy if own_main else origin.cy
    competing = [
        (core.cx, core.cy) for offer_id, core in all_cores.items()
        if offer_id != candidate_id and support.contains_point(core.cx, core.cy)
    ]
    for fact in page.numeric_facts:
        if fact.role != NumericRole.PRICE_MAIN or fact.confidence < .45:
            continue
        if not support.contains_point(fact.bbox.cx, fact.bbox.cy):
            continue
        if math.hypot(fact.bbox.cx - current_x, fact.bbox.cy - current_y) <= max(8.0, max(fact.bbox.width, fact.bbox.height) * .75):
            continue
        if not any(math.hypot(x - fact.bbox.cx, y - fact.bbox.cy) < 8 for x, y in competing):
            competing.append((fact.bbox.cx, fact.bbox.cy))
    if not competing:
        return support, False
    partitioned = _partition_container(support, (current_x, current_y), competing)
    if partitioned.width < 8 or partitioned.height < 8:
        return support, True
    return partitioned, True


def _clip_competing_main_prices(region: BBox, essential_core: BBox, own_main: NumericFact | None, page: PageScene) -> BBox:
    """Ensure the final crop cannot retain an unrelated main-price centre."""
    if own_main is None:
        return region
    result = region
    own_x, own_y = own_main.bbox.cx, own_main.bbox.cy
    others = [
        fact for fact in page.numeric_facts
        if fact.role == NumericRole.PRICE_MAIN and fact.id != own_main.id and fact.confidence >= .45
    ]
    for other in sorted(others, key=lambda fact: math.hypot(fact.bbox.cx - own_x, fact.bbox.cy - own_y)):
        if not result.contains_point(other.bbox.cx, other.bbox.cy):
            continue
        dx, dy = other.bbox.cx - own_x, other.bbox.cy - own_y
        proposal = result
        if abs(dx) >= max(1.0, abs(dy)) * 1.2:
            boundary = (own_x + other.bbox.cx) / 2
            proposal = BBox(max(result.x0, boundary), result.top, result.x1, result.bottom) if dx < 0 else BBox(result.x0, result.top, min(result.x1, boundary), result.bottom)
        elif abs(dy) >= max(1.0, abs(dx)) * 1.2:
            boundary = (own_y + other.bbox.cy) / 2
            proposal = BBox(result.x0, max(result.top, boundary), result.x1, result.bottom) if dy < 0 else BBox(result.x0, result.top, result.x1, min(result.bottom, boundary))
        if proposal.contains(essential_core) and proposal.width >= max(8.0, essential_core.width) and proposal.height >= max(8.0, essential_core.height):
            result = proposal
    return result.clip(page.width, page.height)


def _offer_bbox(page: PageScene, candidate: OfferCandidate, objects: list[VisualObject], facts: list[NumericFact], all_cores: dict[str, BBox]) -> tuple[BBox, str]:
    key_objects = [obj for obj in objects if obj.semantic_role != SemanticRole.IMAGE]
    essential_boxes = [obj.bbox for obj in key_objects] + [fact.bbox for fact in facts]
    essential_core = essential_boxes[0] if essential_boxes else (candidate.bbox or BBox(0, 0, page.width, page.height))
    for box in essential_boxes[1:]:
        essential_core = essential_core.union(box)
    own_main = next((fact for fact in facts if fact.role == NumericRole.PRICE_MAIN), None)
    boxes = list(essential_boxes)
    image_boxes = [obj.bbox for obj in objects if obj.semantic_role == SemanticRole.IMAGE and obj.metadata.get("page_fraction", 1) <= .18]
    boxes.extend(image_boxes)
    # InDesign often stores the photographs of several neighbouring offers in
    # one shared raster. Treat that raster as visual support for every nearby
    # semantic nucleus; later graph barriers partition it between the offers.
    support_radius = max(24.0, math.hypot(page.width, page.height) * .085)
    for obj in page.objects:
        fraction = obj.metadata.get("page_fraction", 0)
        if obj.semantic_role != SemanticRole.IMAGE or not (.04 <= fraction <= .38):
            continue
        overlap_x = max(0.0, min(obj.bbox.x1, essential_core.x1) - max(obj.bbox.x0, essential_core.x0))
        overlap_y = max(0.0, min(obj.bbox.bottom, essential_core.bottom) - max(obj.bbox.top, essential_core.top))
        projected = overlap_x >= min(obj.bbox.width, essential_core.width) * .12 or overlap_y >= min(obj.bbox.height, essential_core.height) * .12
        if projected and obj.bbox.distance(essential_core) <= support_radius:
            local_support, _ = _partition_visual_support(
                obj.bbox, essential_core, essential_core, own_main,
                page, all_cores, candidate.id,
            )
            boxes.append(local_support)
    core = boxes[0] if boxes else BBox(0, 0, page.width, page.height)
    for box in boxes[1:]:
        core = core.union(box)
    origin = all_cores.get(candidate.id, essential_core)
    centers = [(offer_id, box.cx, box.cy) for offer_id, box in all_cores.items()]
    container, contained_offers = _smallest_container(page, core, centers)
    pad = max(page.width, page.height) * .012
    pad_left = pad_right = pad_top = pad_bottom = pad
    # Do not add margin towards a neighbouring offer in the same visual row
    # or column.  This keeps a complete edge product while avoiding a sliver
    # of its neighbour; the decision is derived from the offer graph, not a
    # fixed page grid.
    for offer_id, other in all_cores.items():
        if offer_id == candidate.id:
            continue
        dx, dy = other.cx - origin.cx, other.cy - origin.cy
        if abs(dx) >= abs(dy) * 1.5:
            if dx < 0:
                pad_left = 0.0
                core = BBox(max(core.x0, essential_core.x0 - pad), core.top, core.x1, core.bottom)
            else:
                pad_right = 0.0
                core = BBox(core.x0, core.top, min(core.x1, essential_core.x1 + pad), core.bottom)
        elif abs(dy) >= abs(dx) * 1.5:
            if dy < 0: pad_top = 0.0
            else: pad_bottom = 0.0
    fallback = BBox(core.x0 - pad_left, core.top - pad_top, core.x1 + pad_right, core.bottom + pad_bottom).clip(page.width, page.height)

    def apply_graph_barriers(region: BBox) -> BBox:
        """Clip any proposed region by neighbouring offer axes.

        This is applied after container selection as well: an InDesign image
        frame can legitimately contain several offers and is therefore not a
        sufficient boundary on its own.
        """
        result = region
        for offer_id, other in all_cores.items():
            if offer_id == candidate.id:
                continue
            dx, dy = other.cx - origin.cx, other.cy - origin.cy
            proposal = result
            if abs(dx) >= abs(dy) * 1.5:
                boundary = (origin.cx + other.cx) / 2
                if dx < 0:
                    boundary = max(boundary, essential_core.x0 - pad)
                    proposal = BBox(max(result.x0, boundary), result.top, result.x1, result.bottom)
                else:
                    boundary = min(boundary, essential_core.x1 + pad)
                    proposal = BBox(result.x0, result.top, min(result.x1, boundary), result.bottom)
            elif abs(dy) >= abs(dx) * 1.5:
                boundary = (origin.cy + other.cy) / 2
                if dy < 0:
                    boundary = max(boundary, essential_core.top - pad)
                    proposal = BBox(result.x0, max(result.top, boundary), result.x1, result.bottom)
                else:
                    boundary = min(boundary, essential_core.bottom + pad)
                    proposal = BBox(result.x0, result.top, result.x1, min(result.bottom, boundary))
            if proposal.contains(essential_core) and proposal.width >= max(8, essential_core.width) and proposal.height >= max(8, essential_core.height):
                result = proposal
        if core.bottom >= page.height - pad * 4 and result.bottom > core.bottom:
            result = BBox(result.x0, result.top, result.x1, core.bottom)
        result = result.clip(page.width, page.height)
        result = _clip_competing_main_prices(result, essential_core, own_main, page)
        result = _clip_to_header_footer(result, essential_core, page)
        return result.clip(page.width, page.height)

    if container and container.area <= page.width * page.height * .72:
        if contained_offers <= 1:
            return apply_graph_barriers(container.clip(page.width, page.height)), "container_unique"
        other_centers = [(x, y) for offer_id, x, y in centers if offer_id != candidate.id]
        partitioned = _partition_container(container, (origin.cx, origin.cy), other_centers).clip(page.width, page.height)
        if partitioned.contains(essential_core) and partitioned.width >= max(8, essential_core.width) and partitioned.height >= max(8, essential_core.height):
            return apply_graph_barriers(partitioned), "container_partage"
        return apply_graph_barriers(fallback), "limites_ambigues"
    region = fallback
    # Competing cores act as soft barriers only when their projections overlap.
    for offer_id, other in all_cores.items():
        if offer_id == candidate.id:
            continue
        dx, dy = other.cx - origin.cx, other.cy - origin.cy
        if abs(dx) >= abs(dy) * 1.5:
            boundary = (origin.cx + other.cx) / 2
            if other.cx < origin.cx:
                region = BBox(max(region.x0, boundary), region.top, region.x1, region.bottom)
            else:
                region = BBox(region.x0, region.top, min(region.x1, boundary), region.bottom)
        elif abs(dy) >= abs(dx) * 1.5:
            boundary = (origin.cy + other.cy) / 2
            if other.cy < origin.cy:
                region = BBox(region.x0, max(region.top, boundary), region.x1, region.bottom)
            else:
                region = BBox(region.x0, region.top, region.x1, min(region.bottom, boundary))
    snapped = _snap_to_separators(region, page).clip(page.width, page.height)
    if not snapped.contains(essential_core) or snapped.width < max(8, essential_core.width) or snapped.height < max(8, essential_core.height):
        return apply_graph_barriers(fallback), "limites_ambigues"
    return apply_graph_barriers(snapped), "graphe_spatial"


def _contamination(region: BBox, own: set[str], page: PageScene, assignment_owner: dict[str, str]) -> float:
    foreign_area = 0.0
    own_area = 0.0
    for obj in page.objects:
        if obj.id not in assignment_owner or obj.raw_type not in {"line", "image"}:
            continue
        overlap = region.intersection_area(obj.bbox)
        if obj.id in own:
            own_area += overlap
        else:
            foreign_area += overlap
    return foreign_area / max(1.0, own_area + foreign_area)


def _assemble(page: PageScene, candidates: list[OfferCandidate], graph: SpatialGraph, style) -> list[Offer]:
    object_map = page.object_by_id()
    fact_map = {fact.id: fact for fact in page.numeric_facts}
    preliminary: dict[str, BBox] = {}
    for candidate in candidates:
        main_boxes = [fact_map[fact].bbox for fact in candidate.numeric_ids if fact in fact_map and fact_map[fact].role == NumericRole.PRICE_MAIN]
        semantic_boxes = [
            object_map[obj].bbox for obj in candidate.object_ids
            if obj in object_map and object_map[obj].semantic_role in {SemanticRole.PRODUCT_TEXT, SemanticRole.BRAND, SemanticRole.QUANTITY}
        ]
        boxes = main_boxes or semantic_boxes
        if not boxes:
            boxes = [candidate.bbox] if candidate.bbox else []
        if boxes:
            core = boxes[0]
            for box in boxes[1:]: core = core.union(box)
            preliminary[candidate.id] = core
    assignment_owner = {obj_id: candidate.id for candidate in candidates for obj_id in candidate.object_ids}
    offers = []
    for candidate in candidates:
        objects = [object_map[obj_id] for obj_id in candidate.object_ids if obj_id in object_map]
        facts = [fact_map[fact_id] for fact_id in candidate.numeric_ids if fact_id in fact_map]
        main = next((fact for fact in facts if fact.role == NumericRole.PRICE_MAIN), None)
        product = _pick_product(objects, main, style)
        brands = [_brand(obj.text) for obj in objects if obj.semantic_role == SemanticRole.BRAND]
        arabic = [obj.text for obj in objects if obj.semantic_role == SemanticRole.ARABIC_TEXT]
        quantities = [obj.text for obj in objects if obj.semantic_role == SemanticRole.QUANTITY]
        models = [obj.text for obj in objects if obj.semantic_role == SemanticRole.MODEL]
        technical = [obj.text for obj in objects if obj.semantic_role == SemanticRole.TECHNICAL_SPEC]
        cashback = next((fact for fact in facts if fact.role == NumericRole.CASHBACK), None)
        percentage = next((fact for fact in facts if fact.role == NumericRole.DISCOUNT), None)
        credit = next((fact for fact in facts if fact.role == NumericRole.CREDIT_PAYMENT), None)
        basis_obj = next((obj for obj in objects if obj.semantic_role == SemanticRole.PRICE_BASIS), None)
        region, crop_mode = _offer_bbox(page, candidate, objects, facts, preliminary)
        contamination = _contamination(region, set(candidate.object_ids), page, assignment_owner)
        components = [main.confidence if main else .25, .92 if product else .3, max(.15, 1 - contamination)]
        if brands: components.append(.86)
        confidence = math.prod(components) ** (1 / len(components))
        contradictions = list(candidate.contradictions)
        review = []
        if not product: review.append("désignation absente")
        if not main: review.append("prix principal absent")
        if contamination > .12: review.append("contamination avec une offre voisine")
        if region.width < page.width * .04 or region.height < page.height * .045: review.append("limites d’offre instables")
        offers.append(Offer(
            id=uuid.uuid4().hex[:10], page=page.number, bbox=region,
            object_ids=candidate.object_ids, image_ids=[obj.id for obj in objects if obj.semantic_role == SemanticRole.IMAGE],
            product_name=product or "Produit à vérifier", arabic_name=" ".join(dict.fromkeys(arabic)),
            brand=next((brand for brand in brands if brand), ""), model=models[0] if models else "",
            quantity=quantities[-1] if quantities else "", main_price=(main.value + " DT") if main else "",
            percentage=(percentage.value + " %") if percentage else "",
            promotion=parse_promotion(objects, (cashback.value + " DT versés") if cashback else ""),
            cashback=(cashback.value + " DT versés") if cashback else "", price_basis=basis_obj.text if basis_obj else "",
            credit_payment=(credit.value + " DT") if credit else "", technical_specs=technical,
            confidence=confidence, evidence=candidate.evidence, contradictions=contradictions,
            review_reasons=review, crop_mode=crop_mode,
        ))
    return offers


def resolve_document_offers(document: DocumentScene) -> list[Offer]:
    offers = []
    for page in document.pages:
        graph = build_spatial_graph(page, document.style)
        candidates = _offer_candidates(page, graph)
        _assign_objects(page, graph, candidates)
        page_offers = _assemble(page, candidates, graph, document.style)
        # Reject isolated numeric ornaments and folio fragments.  The minimum
        # viable region is learned from the catalogue body type size, so this
        # remains independent of page dimensions and catalogue coordinates.
        min_width = max(8.0, document.style.body_font_size * 2.0)
        min_height = max(8.0, document.style.body_font_size * 1.5)
        offers.extend(offer for offer in page_offers if offer.bbox.width >= min_width and offer.bbox.height >= min_height)
    return offers

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot

from ..domain import BBox, CatalogueStyleProfile, NumericFact, NumericRole, PageScene, SemanticRole, VisualObject


@dataclass
class SpatialGraph:
    boxes: dict[str, BBox] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    edges: dict[tuple[str, str], float] = field(default_factory=dict)

    def weight(self, left: str, right: str) -> float:
        if left == right:
            return 1.0
        return self.edges.get(tuple(sorted((left, right))), 0.0)

    def neighbours(self, node: str, minimum: float = 0.0) -> list[tuple[str, float]]:
        result = []
        for (left, right), weight in self.edges.items():
            if weight < minimum:
                continue
            if left == node:
                result.append((right, weight))
            elif right == node:
                result.append((left, weight))
        return sorted(result, key=lambda item: item[1], reverse=True)


def _common_container(left: BBox, right: BBox, containers: list[VisualObject]) -> float:
    enclosing = [obj.bbox for obj in containers if obj.bbox.contains(left, 2) and obj.bbox.contains(right, 2)]
    if not enclosing:
        return 0.0
    smallest = min(enclosing, key=lambda bbox: bbox.area)
    return min(1.0, (left.area + right.area) / max(1.0, smallest.area) * 3 + .25)


def _barrier_penalty(left: BBox, right: BBox, separators: list[VisualObject]) -> float:
    penalty = 0.0
    for separator in separators:
        orientation = separator.metadata.get("orientation")
        if orientation == "vertical" and min(left.cx, right.cx) < separator.bbox.cx < max(left.cx, right.cx):
            penalty = max(penalty, min(1.0, separator.bbox.width / max(1.0, min(left.width, right.width)) * 5 + .25))
        elif orientation == "horizontal" and min(left.cy, right.cy) < separator.bbox.cy < max(left.cy, right.cy):
            penalty = max(penalty, min(1.0, separator.bbox.height / max(1.0, min(left.height, right.height)) * 5 + .25))
    return penalty


def _semantic_bonus(left_kind: str, right_kind: str) -> float:
    pair = {left_kind, right_kind}
    if "PRICE_MAIN" in pair and pair & {SemanticRole.PRODUCT_TEXT.value, SemanticRole.BRAND.value, SemanticRole.IMAGE.value}:
        return .28
    if pair == {SemanticRole.PRODUCT_TEXT.value, SemanticRole.BRAND.value}:
        return .24
    if SemanticRole.PROMOTION.value in pair and "PRICE_MAIN" in pair:
        return .22
    if SemanticRole.HEADER_FOOTER.value in pair:
        return -.75
    return 0.0


def build_spatial_graph(page: PageScene, style: CatalogueStyleProfile) -> SpatialGraph:
    graph = SpatialGraph()
    containers = [obj for obj in page.objects if obj.semantic_role == SemanticRole.CONTAINER]
    nodes: list[tuple[str, BBox, str]] = []
    for obj in page.objects:
        if obj.raw_type == "line" and obj.semantic_role != SemanticRole.RAW_TEXT:
            nodes.append((obj.id, obj.bbox, obj.semantic_role.value))
        elif obj.semantic_role == SemanticRole.IMAGE and obj.metadata.get("page_fraction", 1) <= .55 and obj.bbox.width >= style.body_font_size * 2 and obj.bbox.height >= style.body_font_size * 2:
            nodes.append((obj.id, obj.bbox, SemanticRole.IMAGE.value))
    for fact in page.numeric_facts:
        nodes.append((fact.id, fact.bbox, fact.role.value))
    for node_id, bbox, kind in nodes:
        graph.boxes[node_id] = bbox
        graph.kinds[node_id] = kind
    diagonal = hypot(page.width, page.height)
    local_scale = max(style.body_font_size * 8, diagonal * .075)
    for index, (left_id, left, left_kind) in enumerate(nodes):
        for right_id, right, right_kind in nodes[index + 1:]:
            distance = left.distance(right)
            if distance > diagonal * .34:
                continue
            proximity = exp(-distance / max(1.0, local_scale))
            horizontal_overlap = left.intersection_area(BBox(right.x0, left.top, right.x1, left.bottom)) / max(1.0, min(left.area, right.area))
            vertical_overlap = left.intersection_area(BBox(left.x0, right.top, left.x1, right.bottom)) / max(1.0, min(left.area, right.area))
            align_x = exp(-abs(left.cx - right.cx) / max(1.0, (left.width + right.width) * .75))
            align_y = exp(-abs(left.cy - right.cy) / max(1.0, (left.height + right.height) * 1.4))
            container = _common_container(left, right, containers)
            barrier = _barrier_penalty(left, right, page.separators)
            semantic = _semantic_bonus(left_kind, right_kind)
            price_related_roles = {
                SemanticRole.PRODUCT_TEXT.value, SemanticRole.BRAND.value,
                SemanticRole.QUANTITY.value, SemanticRole.ARABIC_TEXT.value,
            }
            price_product = "PRICE_MAIN" in {left_kind, right_kind} and bool({left_kind, right_kind} & price_related_roles)
            directional = 0.0
            if price_product:
                price_box, product_box = (left, right) if left_kind == "PRICE_MAIN" else (right, left)
                directional += .3 * align_x
                if price_box.cy <= product_box.cy + product_box.height:
                    directional += .12
                if abs(price_box.cx - product_box.cx) > max(price_box.width, product_box.width) * 1.25:
                    directional -= .28
            weight = .31 * proximity + .18 * align_x + .08 * align_y + .1 * max(horizontal_overlap, vertical_overlap) + .17 * container + semantic + directional - .42 * barrier
            if weight > .08:
                graph.edges[tuple(sorted((left_id, right_id)))] = min(1.0, weight)
    return graph

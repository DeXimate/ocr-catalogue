from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

from ..domain import CatalogueStyleProfile, DocumentScene, NumericRole, SemanticRole


def _normalise_repeated(text: str) -> str:
    value = re.sub(r"\d+", "#", text.upper())
    value = re.sub(r"\s+", " ", value).strip(" .,:;-–—")
    return value


def _modes(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    tolerance = statistics.median(gaps) * .42 if gaps else .04
    tolerance = max(.012, min(.09, tolerance))
    clusters: list[list[float]] = []
    for value in ordered:
        if clusters and abs(statistics.mean(clusters[-1]) - value) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [round(statistics.mean(cluster), 4) for cluster in clusters if len(cluster) >= 2]


def _repeated_noise(document: DocumentScene) -> set[str]:
    occurrences: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for page in document.pages:
        for obj in page.objects:
            if obj.raw_type != "line" or len(obj.text.strip()) < 4:
                continue
            occurrences[_normalise_repeated(obj.text)].append((page.number, obj.bbox.cy / page.height))
    minimum_pages = max(2, round(len(document.pages) * .22))
    noise = set()
    for text, positions in occurrences.items():
        pages = {page for page, _ in positions}
        ys = [position for _, position in positions]
        if len(pages) >= minimum_pages and (statistics.pstdev(ys) <= .035 or statistics.mean(ys) <= .1 or statistics.mean(ys) >= .88):
            noise.add(text)
    return noise


def infer_catalogue_style(document: DocumentScene) -> CatalogueStyleProfile:
    lines = [obj for page in document.pages for obj in page.objects if obj.raw_type == "line" and obj.semantic_role != SemanticRole.HEADER_FOOTER]
    body_sizes = [obj.metadata.get("mean_font_size", obj.font_size) for obj in lines if obj.semantic_role in {SemanticRole.PRODUCT_TEXT, SemanticRole.QUANTITY, SemanticRole.ARABIC_TEXT}]
    body_size = statistics.median(body_sizes) if body_sizes else 8.0
    by_id = {obj.id: obj for page in document.pages for obj in page.objects}
    main_prices = [fact for page in document.pages for fact in page.numeric_facts if fact.role == NumericRole.PRICE_MAIN]
    price_sizes = [max((by_id[source].font_size for source in fact.source_ids if source in by_id), default=fact.bbox.height) for fact in main_prices]
    percentage_sizes = [fact.bbox.height for page in document.pages for fact in page.numeric_facts if fact.role == NumericRole.DISCOUNT]
    price_fonts = Counter(by_id[source].font_name for fact in main_prices for source in fact.source_ids if source in by_id and by_id[source].font_name)
    product_fonts = Counter(obj.font_name for obj in lines if obj.semantic_role == SemanticRole.PRODUCT_TEXT and obj.font_name)
    noise = _repeated_noise(document)
    for page in document.pages:
        for obj in page.objects:
            if obj.raw_type == "line" and _normalise_repeated(obj.text) in noise:
                obj.semantic_role = SemanticRole.HEADER_FOOTER
                obj.semantic_confidence = max(obj.semantic_confidence, .92)
            elif obj.raw_type == "line" and obj.font_size >= body_size * 2.8 and obj.bbox.width >= page.width * .24:
                obj.semantic_role = SemanticRole.HEADER_FOOTER
                obj.semantic_confidence = max(obj.semantic_confidence, .82)
    price_size = statistics.median(price_sizes) if price_sizes else body_size * 2
    x_modes = _modes([fact.bbox.cx / page.width for page in document.pages for fact in page.numeric_facts if fact.role == NumericRole.PRICE_MAIN])
    y_modes = _modes([fact.bbox.cy / page.height for page in document.pages for fact in page.numeric_facts if fact.role == NumericRole.PRICE_MAIN])
    page_profiles = {}
    for page in document.pages:
        text_count = sum(obj.raw_type == "line" for obj in page.objects)
        image_count = sum(obj.raw_type == "image" and obj.metadata.get("page_fraction", 1) < .65 for obj in page.objects)
        price_count = sum(fact.role == NumericRole.PRICE_MAIN for fact in page.numeric_facts)
        container_count = sum(obj.raw_type == "container" for obj in page.objects)
        density = price_count / max(1, page.width * page.height) * 100000
        page_profiles[page.number] = {
            "text_count": text_count, "image_count": image_count, "price_count": price_count,
            "container_count": container_count, "offer_density": density,
            "free_composition": price_count <= max(3, statistics.median([sum(f.role == NumericRole.PRICE_MAIN for f in p.numeric_facts) for p in document.pages]) * .35),
        }
    profile = CatalogueStyleProfile(
        body_font_size=body_size, price_font_size=price_size,
        percentage_font_size=statistics.median(percentage_sizes) if percentage_sizes else body_size,
        price_fonts=[name for name, _ in price_fonts.most_common(5)],
        product_fonts=[name for name, _ in product_fonts.most_common(5)],
        repeated_noise=noise, alignment_modes_x=x_modes, alignment_modes_y=y_modes,
        page_profiles=page_profiles, evidence_count=len(main_prices),
    )
    # Second pass: adapt price confidence to the document's own visual grammar.
    for page in document.pages:
        discounts = [fact for fact in page.numeric_facts if fact.role == NumericRole.DISCOUNT]
        for fact in page.numeric_facts:
            if fact.role != NumericRole.PRICE_MAIN:
                continue
            sources = [by_id[source] for source in fact.source_ids if source in by_id]
            size = max((obj.font_size for obj in sources), default=fact.bbox.height)
            size_fit = min(1.0, size / max(body_size * 1.35, price_size * .72))
            font_fit = 1.0 if any(obj.font_name in profile.price_fonts for obj in sources) else .55
            nearby_discount = any(discount.bbox.distance(fact.bbox) <= max(fact.bbox.height, discount.bbox.height) * 2.8 for discount in discounts)
            fact.confidence = min(.99, .48 + .25 * size_fit + .12 * font_fit + (.11 if nearby_discount else 0))
            fact.evidence.extend(["style_prix_catalogue", "pourcentage_local"] if nearby_discount else ["style_prix_catalogue"])
    document.style = profile
    return profile

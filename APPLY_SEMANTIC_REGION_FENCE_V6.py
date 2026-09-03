from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CLASSIFIER = ROOT / "ocr_catalogue" / "semantics" / "classifier.py"
RESOLVER = ROOT / "ocr_catalogue" / "offers" / "resolver.py"


def replace_function(text: str, function_name: str, next_function_name: str, replacement: str) -> str:
    start_marker = f"def {function_name}("
    end_marker = f"\ndef {next_function_name}("
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"fonction {function_name} introuvable")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"fonction {next_function_name} introuvable après {function_name}")
    return text[:start] + replacement.rstrip() + "\n" + text[end:]


ASSEMBLE = r'''def _assemble(
    page: PageScene,
    candidates: list[OfferCandidate],
    graph: SpatialGraph,
    style,
    region_solutions: dict[str, RegionSolution],
) -> list[Offer]:
    object_map = page.object_by_id()
    fact_map = {fact.id: fact for fact in page.numeric_facts}
    assignment_owner = {
        obj_id: candidate.id
        for candidate in candidates
        for obj_id in candidate.object_ids
    }
    offers = []

    for candidate in candidates:
        solution = region_solutions[candidate.id]
        region, crop_mode = solution.region, solution.mode

        local_object_ids = [
            obj_id
            for obj_id in candidate.object_ids
            if obj_id in object_map
            and region.contains_point(
                object_map[obj_id].bbox.cx,
                object_map[obj_id].bbox.cy,
                2.0,
            )
        ]
        local_numeric_ids = [
            fact_id
            for fact_id in candidate.numeric_ids
            if fact_id in fact_map
            and region.contains_point(
                fact_map[fact_id].bbox.cx,
                fact_map[fact_id].bbox.cy,
                2.0,
            )
        ]

        objects = [object_map[obj_id] for obj_id in local_object_ids]
        facts = [fact_map[fact_id] for fact_id in local_numeric_ids]

        main = next(
            (fact for fact in facts if fact.role == NumericRole.PRICE_MAIN),
            None,
        )
        if main is None:
            main = next(
                (
                    fact_map[fact_id]
                    for fact_id in candidate.numeric_ids
                    if fact_id in fact_map
                    and fact_map[fact_id].role == NumericRole.PRICE_MAIN
                ),
                None,
            )
            if main is not None and main.id not in local_numeric_ids:
                facts.append(main)
                local_numeric_ids.append(main.id)

        product = _pick_product(objects, main, style)
        brands = [
            _brand(obj.text)
            for obj in objects
            if obj.semantic_role == SemanticRole.BRAND
        ]
        arabic = [
            obj.text
            for obj in objects
            if obj.semantic_role == SemanticRole.ARABIC_TEXT
        ]

        quantity, technical = _format_and_characteristics(product, objects, facts)
        model = _extract_model(objects)

        if model:
            cleaned_technical = []
            for value in technical:
                cleaned = re.sub(
                    rf"(?<![A-Z0-9]){re.escape(model)}(?![A-Z0-9])",
                    "",
                    value,
                    flags=re.I,
                )
                cleaned = re.sub(r"\s*[-–—]\s*", " - ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip(" -•")
                if cleaned:
                    cleaned_technical.append(cleaned)
            technical = _dedupe_text(cleaned_technical)

        variant_prices = _variant_prices(facts)
        cashback = next((fact for fact in facts if fact.role == NumericRole.CASHBACK), None)
        percentage = next((fact for fact in facts if fact.role == NumericRole.DISCOUNT), None)
        credit = next((fact for fact in facts if fact.role == NumericRole.CREDIT_PAYMENT), None)
        basis_obj = next((obj for obj in objects if obj.semantic_role == SemanticRole.PRICE_BASIS), None)

        contamination = _contamination(
            region,
            set(local_object_ids),
            page,
            assignment_owner,
        )
        components = [
            main.confidence if main else .25,
            .92 if product else .3,
            max(.15, 1 - contamination),
        ]
        if brands:
            components.append(.86)
        confidence = math.prod(components) ** (1 / len(components))

        contradictions = list(candidate.contradictions)
        review = []
        if not product:
            review.append("désignation absente")
        if not main:
            review.append("prix principal absent")
        if contamination > .12 or solution.quality.get("foreign_offer_contamination", 0) >= .5:
            review.append("contamination avec une offre voisine")
        if not solution.quality.get("accepted", False):
            review.append("région d’offre à contrôler")
        if region.width < page.width * .04 or region.height < page.height * .045:
            review.append("limites d’offre instables")

        offers.append(Offer(
            id=uuid.uuid4().hex[:10],
            page=page.number,
            bbox=region,
            object_ids=local_object_ids,
            image_ids=[obj.id for obj in objects if obj.semantic_role == SemanticRole.IMAGE],
            product_name=product or "Produit à vérifier",
            arabic_name=" ".join(dict.fromkeys(arabic)),
            brand=next((brand for brand in brands if brand), ""),
            model=model,
            variant=variant_prices,
            quantity=quantity,
            main_price=(main.value + " DT") if main else "",
            percentage=(percentage.value + " %") if percentage else "",
            promotion=parse_promotion(
                objects,
                (cashback.value + " DT versés") if cashback else "",
            ),
            cashback=(cashback.value + " DT versés") if cashback else "",
            price_basis=basis_obj.text if basis_obj else "",
            credit_payment=(credit.value + " DT") if credit else "",
            technical_specs=technical,
            confidence=confidence,
            evidence=candidate.evidence,
            contradictions=contradictions,
            review_reasons=review,
            crop_mode=crop_mode,
            safe_bbox=solution.safe_region.as_list(),
            region_quality=solution.quality,
        ))

    return offers
'''


def patch_classifier() -> None:
    text = CLASSIFIER.read_text(encoding="utf-8")

    if "MONEY_ONLY = re.compile" not in text:
        marker = 'PRICE_COMPACT = re.compile(r"^(\\d{1,4})\\s*[,\\.]\\s*(\\d{3})\\s*(?:DT)?$", re.I)\n'
        addition = 'MONEY_ONLY = re.compile(r"^\\s*(?:\\+?\\d{1,4}(?:[,.]\\d{3})?\\s*D+T+\\s*){1,6}$", re.I)\n'
        if marker not in text:
            raise RuntimeError("PRICE_COMPACT introuvable")
        text = text.replace(marker, marker + addition, 1)

    if 'elif MONEY_ONLY.fullmatch(text):' not in text:
        marker = '''        elif ARABIC.search(text) and not re.search(r"[A-Za-zÀ-ÿ]", text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.ARABIC_TEXT, .93
'''
        if marker not in text:
            raise RuntimeError("point d'insertion MONEY_ONLY introuvable")
        text = text.replace(
            marker,
            marker + '''        elif MONEY_ONLY.fullmatch(text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.RAW_TEXT, .98
''',
            1,
        )

    old_model = 'elif MODEL.search(text) and not re.search(r"\\s", text.strip()):'
    new_model = 'elif MODEL.search(text) and not re.search(r"\\s", text.strip()) and not MONEY_ONLY.fullmatch(text):'
    if old_model in text:
        text = text.replace(old_model, new_model, 1)

    old_product = 'and not re.search(r"\\bDT\\b", text, re.I):'
    new_product = 'and not re.search(r"\\d+\\s*D+T+\\b|\\bD+T+\\s*\\d+", text, re.I) and not MONEY_ONLY.fullmatch(text):'
    if old_product in text:
        text = text.replace(old_product, new_product, 1)

    if new_product not in text:
        raise RuntimeError("garde monétaire PRODUCT_TEXT introuvable")

    compile(text, str(CLASSIFIER), "exec")
    CLASSIFIER.write_text(text, encoding="utf-8")


def patch_resolver() -> None:
    text = RESOLVER.read_text(encoding="utf-8")
    text = replace_function(text, "_assemble", "resolve_document_offers", ASSEMBLE)
    compile(text, str(RESOLVER), "exec")
    RESOLVER.write_text(text, encoding="utf-8")


def main() -> None:
    if not CLASSIFIER.exists() or not RESOLVER.exists():
        raise RuntimeError("classifier.py ou resolver.py introuvable")

    patch_classifier()
    patch_resolver()

    print("SEMANTIC REGION FENCE V7 APPLIQUE")
    print("Aucun re.sub n'est utilise pour injecter une fonction.")
    print("Corrections:")
    print(" - garantie voisine exclue hors du crop final")
    print(" - 89DT 51DT ne peut plus devenir Produit")
    print(" - 148DT ne peut plus devenir Modele")
    print(" - modele retire des Caracteristiques")
    print(r"Etape suivante: .\test.ps1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        raise

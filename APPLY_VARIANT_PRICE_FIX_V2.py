from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

FILES = {
    "domain": ROOT / "ocr_catalogue" / "domain" / "models.py",
    "classifier": ROOT / "ocr_catalogue" / "semantics" / "classifier.py",
    "resolver": ROOT / "ocr_catalogue" / "offers" / "resolver.py",
    "models": ROOT / "ocr_catalogue" / "models.py",
    "pipeline": ROOT / "ocr_catalogue" / "pipeline.py",
    "index": ROOT / "static" / "index.html",
    "appjs": ROOT / "static" / "app.js",
    "exporter": ROOT / "ocr_catalogue" / "exporter.py",
    "tests_offer": ROOT / "tests" / "test_offer_engine.py",
    "tests_core": ROOT / "tests" / "test_core.py",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: motif attendu 1 fois, trouvé {count} fois")
    return text.replace(old, new, 1)


def patch_domain() -> None:
    path = FILES["domain"]
    text = path.read_text(encoding="utf-8")
    # Idempotent: the local file may already contain the enum value even when
    # the previous migration script did not finish.
    if 'VARIANT_PRICE = "VARIANT_PRICE"' in text:
        return
    marker = '    PRICE_MAIN = "PRICE_MAIN"'
    if marker not in text:
        raise RuntimeError("NumericRole.PRICE_MAIN introuvable dans domain/models.py")
    text = text.replace(marker, marker + '\n    VARIANT_PRICE = "VARIANT_PRICE"', 1)
    path.write_text(text, encoding="utf-8")


def patch_classifier() -> None:
    path = FILES["classifier"]
    text = path.read_text(encoding="utf-8")

    marker = "def _preceded_by_reference_cue(page: PageScene, bbox: BBox) -> bool:\n"
    helper = r"""def _variant_descriptor(page: PageScene, bbox: BBox) -> str:
    # Recover the descriptor attached to an alternative price from the same visual row.
    words = [obj for obj in page.objects if obj.raw_type == "word"]
    row_tolerance = max(8.0, min(14.0, bbox.height * .42))
    horizontal_window = max(150.0, bbox.height * 12.0)
    left = [
        word for word in words
        if word.bbox.x1 <= bbox.x0 + max(3.0, bbox.width * .06)
        and 0 <= bbox.x0 - word.bbox.x1 <= horizontal_window
        and abs(word.bbox.cy - bbox.cy) <= max(row_tolerance, word.font_size * 1.25)
    ]
    left.sort(key=lambda word: word.bbox.x0)
    if not left:
        return ""

    at_indexes = [
        index for index, word in enumerate(left)
        if word.text.strip().lower().strip(":;") in {"à", "a"}
        and bbox.x0 - word.bbox.x1 <= max(60.0, bbox.height * 4.0)
    ]
    if not at_indexes:
        return ""
    at_index = at_indexes[-1]

    tokens: list[str] = []
    boundary_found = False
    for index in range(at_index - 1, -1, -1):
        raw = left[index].text.strip()
        token = raw.strip(" :;-")
        normalized = token.lower()
        if normalized in {"en", "et", "existe"}:
            boundary_found = True
            break
        if not token:
            continue
        compact = token.upper().replace(" ", "")
        if compact in {"DT", "DTT"} or "%" in token:
            break
        if re.fullmatch(r"\d{1,4}[,.]\d{3}", token):
            break
        tokens.append(token)
        if len(tokens) >= 10:
            break

    if not boundary_found or not tokens:
        return ""
    descriptor = re.sub(r"\s+", " ", " ".join(reversed(tokens))).strip(" -,:;")
    if not descriptor or re.fullmatch(r"\d+", descriptor):
        return ""
    return descriptor


"""
    text = replace_once(text, marker, helper + marker, "insert variant descriptor")

    old = "    if _preceded_by_reference_cue(page, bbox):\n        return NumericRole.TECHNICAL_SPEC, .9, [\"prix_variante_non_exporte\"]\n    return NumericRole.PRICE_MAIN, .72, [\"expression_dt_complete\"]"
    new = "    variant = _variant_descriptor(page, bbox)\n    if variant:\n        return NumericRole.VARIANT_PRICE, .95, [\"prix_variante\", f\"variant_label:{variant}\"]\n    return NumericRole.PRICE_MAIN, .72, [\"expression_dt_complete\"]"
    text = replace_once(text, old, new, "price role variant classification")

    old = "    role_priority = {NumericRole.CASHBACK: 3, NumericRole.CREDIT_PAYMENT: 3, NumericRole.PRICE_MAIN: 1}"
    new = "    role_priority = {\n        NumericRole.CASHBACK: 4,\n        NumericRole.CREDIT_PAYMENT: 4,\n        NumericRole.VARIANT_PRICE: 2,\n        NumericRole.PRICE_MAIN: 1,\n    }"
    text = replace_once(text, old, new, "variant price priority")

    path.write_text(text, encoding="utf-8")


def patch_resolver() -> None:
    path = FILES["resolver"]
    text = path.read_text(encoding="utf-8")

    marker = "def _format_and_characteristics(\n"
    helper = r"""def _variant_prices(facts: list[NumericFact]) -> str:
    # Render alternative commercial variants without mixing them with specs.
    items: list[str] = []
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        legacy_variant = "prix_variante_non_exporte" in fact.evidence
        if fact.role != NumericRole.VARIANT_PRICE and not legacy_variant:
            continue
        label = next(
            (evidence.split(":", 1)[1].strip() for evidence in fact.evidence if evidence.startswith("variant_label:")),
            "",
        )
        value = fact.value.strip()
        if not value:
            continue
        key = (re.sub(r"\W+", "", label).lower(), value)
        if key in seen:
            continue
        seen.add(key)
        items.append(f"{label} : {value} DT" if label else f"Autre variante : {value} DT")
    return " • ".join(items)


"""
    text = replace_once(text, marker, helper + marker, "insert variant formatter")

    old = "    for fact in facts:\n        if fact.role not in {\n            NumericRole.POWER,"
    new = "    for fact in facts:\n        # Variant prices belong to their own commercial field, never characteristics.\n        if fact.role == NumericRole.VARIANT_PRICE or \"prix_variante_non_exporte\" in fact.evidence:\n            continue\n        if fact.role not in {\n            NumericRole.POWER,"
    text = replace_once(text, old, new, "exclude variants from characteristics")

    old = "        quantity, technical = _format_and_characteristics(product, objects, facts)\n        model = _extract_model(objects)\n        cashback = next((fact for fact in facts if fact.role == NumericRole.CASHBACK), None)"
    new = "        quantity, technical = _format_and_characteristics(product, objects, facts)\n        model = _extract_model(objects)\n        variant_prices = _variant_prices(facts)\n        cashback = next((fact for fact in facts if fact.role == NumericRole.CASHBACK), None)"
    text = replace_once(text, old, new, "compute variant prices")

    old = "            brand=next((brand for brand in brands if brand), \"\"), model=model,\n            quantity=quantity, main_price=(main.value + \" DT\") if main else \"\","
    new = "            brand=next((brand for brand in brands if brand), \"\"), model=model, variant=variant_prices,\n            quantity=quantity, main_price=(main.value + \" DT\") if main else \"\","
    text = replace_once(text, old, new, "map variant prices to Offer.variant")

    path.write_text(text, encoding="utf-8")


def patch_models() -> None:
    path = FILES["models"]
    text = path.read_text(encoding="utf-8")
    old = "    quantite: str = \"\"\n    caracteristiques: str = \"\"\n    prix_promo: str = \"\""
    new = "    quantite: str = \"\"\n    caracteristiques: str = \"\"\n    variantes_prix: str = \"\"\n    prix_promo: str = \"\""
    text = replace_once(text, old, new, "Product.variantes_prix")
    path.write_text(text, encoding="utf-8")


def patch_pipeline() -> None:
    path = FILES["pipeline"]
    text = path.read_text(encoding="utf-8")
    old = "        marque=offer.brand, modele=offer.model, quantite=offer.quantity,\n        caracteristiques=\" • \".join(offer.technical_specs),\n        prix_promo=offer.main_price, pourcentage=offer.percentage,"
    new = "        marque=offer.brand, modele=offer.model, quantite=offer.quantity,\n        caracteristiques=\" • \".join(offer.technical_specs), variantes_prix=offer.variant,\n        prix_promo=offer.main_price, pourcentage=offer.percentage,"
    text = replace_once(text, old, new, "pipeline variant prices mapping")
    path.write_text(text, encoding="utf-8")


def patch_index() -> None:
    path = FILES["index"]
    text = path.read_text(encoding="utf-8")
    old = "<th>Modèle</th><th>Format / conditionnement</th><th>Caractéristiques</th><th>Prix promo</th>"
    new = "<th>Modèle</th><th>Format / conditionnement</th><th>Caractéristiques</th><th>Variantes / autres prix</th><th>Prix promo</th>"
    text = replace_once(text, old, new, "UI variant column")
    path.write_text(text, encoding="utf-8")


def patch_appjs() -> None:
    path = FILES["appjs"]
    text = path.read_text(encoding="utf-8")
    old = "${['produit','marque','modele','quantite','caracteristiques','prix_promo','pourcentage','promotion'].map(key=>editableCell(product,key)).join('')}"
    new = "${['produit','marque','modele','quantite','caracteristiques','variantes_prix','prix_promo','pourcentage','promotion'].map(key=>editableCell(product,key)).join('')}"
    text = replace_once(text, old, new, "app variant column")
    path.write_text(text, encoding="utf-8")


def patch_exporter() -> None:
    path = FILES["exporter"]
    text = path.read_text(encoding="utf-8")
    old = "    (\"modele\", \"Modèle\"), (\"quantite\", \"Format / conditionnement\"),\n    (\"caracteristiques\", \"Caractéristiques\"), (\"prix_promo\", \"Prix promo\"),"
    new = "    (\"modele\", \"Modèle\"), (\"quantite\", \"Format / conditionnement\"),\n    (\"caracteristiques\", \"Caractéristiques\"),\n    (\"variantes_prix\", \"Variantes / autres prix\"), (\"prix_promo\", \"Prix promo\"),"
    text = replace_once(text, old, new, "export variant column")

    old = "        \"Format / conditionnement\": 24, \"Caractéristiques\": 48,\n        \"Prix promo\": 16, \"Pourcentage\": 16, \"Promotion\": 28,"
    new = "        \"Format / conditionnement\": 24, \"Caractéristiques\": 48,\n        \"Variantes / autres prix\": 42, \"Prix promo\": 16,\n        \"Pourcentage\": 16, \"Promotion\": 28,"
    text = replace_once(text, old, new, "export variant width")
    path.write_text(text, encoding="utf-8")


def patch_tests_offer() -> None:
    path = FILES["tests_offer"]
    text = path.read_text(encoding="utf-8")

    old = "from ocr_catalogue.offers.resolver import _extract_model, _format_and_characteristics, _merge_product_with_priced_brand, _offer_bbox, _offer_candidates, _partition_container, _pick_product, _reassign_secondary_facts"
    new = "from ocr_catalogue.offers.resolver import _extract_model, _format_and_characteristics, _merge_product_with_priced_brand, _offer_bbox, _offer_candidates, _partition_container, _pick_product, _reassign_secondary_facts, _variant_prices"
    text = replace_once(text, old, new, "test import variant formatter")

    marker = "    def test_classifier_accepts_only_explicit_free_mechanism(self):\n"
    tests = r"""    def test_alternative_price_gets_dedicated_variant_role(self):
        words = [
            word("existe", "Existe", 0, 20, 32, 30, 10),
            word("en", "en", 34, 20, 46, 30, 10),
            word("small", "small", 48, 20, 78, 30, 10),
            word("at", "à", 80, 20, 86, 30, 10),
            word("head", "19", 90, 8, 118, 38, 28),
            word("dt", "DT", 120, 14, 132, 26, 10),
            word("tail", ",900", 132, 23, 160, 38, 14),
        ]
        page = PageScene(1, 300, 500, words + _line_objects(1, words))
        prices = _find_prices(page)
        variant = next(fact for fact in prices if fact.value == "19,900")
        self.assertEqual(variant.role, NumericRole.VARIANT_PRICE)
        self.assertIn("variant_label:small", variant.evidence)

    def test_main_price_is_not_demoted_by_variant_text_below_it(self):
        words = [
            word("main", "25DT", 100, 20, 150, 60, 36),
            word("main-tail", ",900", 148, 44, 180, 62, 16),
            word("existe", "Existe", 70, 72, 100, 82, 8),
            word("en", "en", 102, 72, 112, 82, 8),
            word("small", "small", 114, 72, 140, 82, 8),
            word("at", "à", 142, 72, 148, 82, 8),
            word("variant", "19DT", 150, 68, 178, 86, 14),
            word("variant-tail", ",900", 176, 76, 202, 88, 10),
        ]
        page = PageScene(1, 300, 500, words + _line_objects(1, words))
        prices = _find_prices(page)
        roles = {fact.value: fact.role for fact in prices}
        self.assertEqual(roles["25,900"], NumericRole.PRICE_MAIN)
        self.assertEqual(roles["19,900"], NumericRole.VARIANT_PRICE)

    def test_variant_prices_do_not_leak_into_characteristics(self):
        variant = NumericFact(
            "variant", 1, "19 DT ,900", "19,900", BBox(0, 20, 50, 35),
            NumericRole.VARIANT_PRICE, .95, evidence=["prix_variante", "variant_label:small"],
        )
        retail_format, characteristics = _format_and_characteristics("Couches adulte", [], [variant])
        self.assertEqual(retail_format, "")
        self.assertEqual(characteristics, [])
        self.assertEqual(_variant_prices([variant]), "small : 19,900 DT")

    def test_multiple_variant_prices_keep_their_descriptors(self):
        facts = [
            NumericFact("small", 1, "19 DT ,900", "19,900", BBox(0, 0, 10, 10), NumericRole.VARIANT_PRICE, .95, evidence=["prix_variante", "variant_label:small"]),
            NumericFact("medium", 1, "22 DT ,900", "22,900", BBox(0, 20, 10, 30), NumericRole.VARIANT_PRICE, .95, evidence=["prix_variante", "variant_label:medium"]),
        ]
        self.assertEqual(_variant_prices(facts), "small : 19,900 DT • medium : 22,900 DT")

"""
    text = replace_once(text, marker, tests + marker, "insert variant tests")
    path.write_text(text, encoding="utf-8")


def patch_tests_core() -> None:
    path = FILES["tests_core"]
    text = path.read_text(encoding="utf-8")
    old = "        self.assertIn(\"Pourcentage\", headers)\n        self.assertNotIn(\"Ancien prix\", headers)"
    new = "        self.assertIn(\"Pourcentage\", headers)\n        self.assertIn(\"Variantes / autres prix\", headers)\n        self.assertNotIn(\"Ancien prix\", headers)"
    text = replace_once(text, old, new, "export variant header test")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise RuntimeError("Fichiers introuvables:\n" + "\n".join(missing))

    patch_domain()
    patch_classifier()
    patch_resolver()
    patch_models()
    patch_pipeline()
    patch_index()
    patch_appjs()
    patch_exporter()
    patch_tests_offer()
    patch_tests_core()

    print("VARIANT PRICE FIX APPLIQUE")
    print("Fichiers modifies:")
    for path in FILES.values():
        print(" -", path.relative_to(ROOT))
    print()
    print(r"Lance maintenant: .\test.ps1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        raise

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent

DOMAIN = ROOT / "ocr_catalogue" / "domain" / "models.py"
CLASSIFIER = ROOT / "ocr_catalogue" / "semantics" / "classifier.py"
RESOLVER = ROOT / "ocr_catalogue" / "offers" / "resolver.py"
MODELS = ROOT / "ocr_catalogue" / "models.py"
PIPELINE = ROOT / "ocr_catalogue" / "pipeline.py"
INDEX = ROOT / "static" / "index.html"
APPJS = ROOT / "static" / "app.js"
EXPORTER = ROOT / "ocr_catalogue" / "exporter.py"

VARIANT_HELPER = 'def _variant_descriptor(page: PageScene, bbox: BBox) -> str:\n    """Recover the label attached to an alternative price on the same visual row."""\n    words = [obj for obj in page.objects if obj.raw_type == "word"]\n    left = [\n        word for word in words\n        if word.bbox.x1 <= bbox.x0 + 3.0\n        and 0 <= bbox.x0 - word.bbox.x1 <= max(150.0, bbox.width * 6.0)\n        and abs(word.bbox.cy - bbox.cy) <= max(8.0, word.font_size * 1.35)\n    ]\n    left.sort(key=lambda word: word.bbox.x0)\n    if not left:\n        return ""\n\n    at_indexes = [\n        index for index, word in enumerate(left)\n        if word.text.strip().lower().strip(" :;") in {"à", "a"}\n        and bbox.x0 - word.bbox.x1 <= max(55.0, word.font_size * 7.0)\n    ]\n    if not at_indexes:\n        return ""\n\n    tokens: list[str] = []\n    boundary_found = False\n    for index in range(at_indexes[-1] - 1, -1, -1):\n        token = left[index].text.strip().strip(" :;-")\n        normalized = token.lower()\n        if normalized in {"en", "et", "existe"}:\n            boundary_found = True\n            break\n        if not token:\n            continue\n        if token.upper().replace(" ", "") in {"DT", "DTT"} or "%" in token:\n            break\n        if re.fullmatch(r"\\d{1,4}[,.]\\d{3}", token):\n            break\n        tokens.append(token)\n        if len(tokens) >= 10:\n            break\n\n    if not boundary_found or not tokens:\n        return ""\n    descriptor = re.sub(r"\\s+", " ", " ".join(reversed(tokens))).strip(" -,:;")\n    if not descriptor or re.fullmatch(r"\\d+", descriptor):\n        return ""\n    return descriptor\n\n\n'
PRICE_ROLE = 'def _price_role(page: PageScene, bbox: BBox) -> tuple[NumericRole, float, list[str]]:\n    context = _numeric_context(page, bbox)\n    plus_near = any(\n        obj.raw_type == "word" and obj.text.strip().startswith("+")\n        and obj.bbox.distance(bbox) <= max(12.0, bbox.height * 1.8)\n        for obj in page.objects\n    )\n    if plus_near and re.search(r"VERS[ÉE]S?", context, re.I):\n        return NumericRole.CASHBACK, .96, ["voisinage_verses"]\n    if re.search(r"ACHAT\\s+[ÀA]\\s+CR[ÉE]DIT", context, re.I) and re.search(r"\\bmois\\b", context, re.I):\n        return NumericRole.CREDIT_PAYMENT, .92, ["voisinage_credit"]\n\n    descriptor = _variant_descriptor(page, bbox)\n    if descriptor:\n        return NumericRole.VARIANT_PRICE, .95, ["prix_variante", f"variant_label:{descriptor}"]\n\n    return NumericRole.PRICE_MAIN, .72, ["expression_dt_complete"]\n\n\n'
FORMAT_FUNC = 'def _format_and_characteristics(\n    product: str,\n    objects: list[VisualObject],\n    facts: list[NumericFact],\n) -> tuple[str, list[str]]:\n    quantities = [obj.text for obj in objects if obj.semantic_role == SemanticRole.QUANTITY]\n    appliance = _is_appliance_offer(product, objects)\n\n    characteristics: list[str] = []\n    for obj in objects:\n        if obj.semantic_role != SemanticRole.TECHNICAL_SPEC:\n            continue\n        text = obj.text.strip()\n        if not text or _CREDIT_ONLY.search(text):\n            continue\n        if appliance or _CHARACTERISTIC_CUE.search(text):\n            characteristics.append(text)\n\n    for fact in facts:\n        # Alternative commercial prices have their own field.\n        if fact.role == NumericRole.VARIANT_PRICE or "prix_variante_non_exporte" in fact.evidence:\n            continue\n        if fact.role not in {\n            NumericRole.POWER,\n            NumericRole.CAPACITY,\n            NumericRole.DIMENSION,\n            NumericRole.DURATION,\n            NumericRole.TECHNICAL_SPEC,\n        }:\n            continue\n        text = fact.text.strip()\n        if not text or _CREDIT_ONLY.search(text):\n            continue\n        if fact.role == NumericRole.DURATION and not re.search(r"garantie", text, re.I):\n            continue\n        characteristics.append(text)\n\n    if appliance:\n        characteristics.extend(quantities)\n        retail_format = ""\n    else:\n        retail_format = quantities[-1] if quantities else ""\n\n    return retail_format, _dedupe_text(characteristics)\n\n\n'
VARIANT_FORMATTER = 'def _variant_prices(facts: list[NumericFact]) -> str:\n    """Render alternative prices without mixing them with technical specs."""\n    items: list[str] = []\n    seen: set[tuple[str, str]] = set()\n\n    for fact in facts:\n        legacy_variant = "prix_variante_non_exporte" in fact.evidence\n        if fact.role != NumericRole.VARIANT_PRICE and not legacy_variant:\n            continue\n\n        label = next(\n            (\n                evidence.split(":", 1)[1].strip()\n                for evidence in fact.evidence\n                if evidence.startswith("variant_label:")\n            ),\n            "",\n        )\n        value = fact.value.strip()\n        if not value:\n            continue\n\n        key = (re.sub(r"\\W+", "", label).lower(), value)\n        if key in seen:\n            continue\n        seen.add(key)\n\n        items.append(\n            f"{label} : {value} DT"\n            if label\n            else f"Autre variante : {value} DT"\n        )\n\n    return " • ".join(items)\n\n\n'


def replace_function(text: str, name: str, replacement: str, next_name: str) -> str:
    pattern = rf"def {re.escape(name)}\(.*?(?=\ndef {re.escape(next_name)}\()"
    updated, count = re.subn(pattern, replacement.rstrip() + "\n\n", text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"fonction {name} introuvable")
    return updated


def ensure_domain() -> None:
    text = DOMAIN.read_text(encoding="utf-8")
    if 'VARIANT_PRICE = "VARIANT_PRICE"' not in text:
        text, count = re.subn(
            r'(?m)^(\s*)PRICE_MAIN\s*=\s*"PRICE_MAIN"\s*$',
            lambda m: m.group(0) + f'\n{m.group(1)}VARIANT_PRICE = "VARIANT_PRICE"',
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("NumericRole.PRICE_MAIN introuvable")
    DOMAIN.write_text(text, encoding="utf-8")


def patch_classifier() -> None:
    text = CLASSIFIER.read_text(encoding="utf-8")

    if "def _variant_descriptor(" not in text:
        marker = "def _preceded_by_reference_cue(page: PageScene, bbox: BBox) -> bool:\n"
        if marker not in text:
            raise RuntimeError("_preceded_by_reference_cue introuvable")
        text = text.replace(marker, VARIANT_HELPER + marker, 1)

    text = replace_function(text, "_price_role", PRICE_ROLE, "_find_prices")

    text, count = re.subn(
        r"role_priority\s*=\s*\{[^\n]*NumericRole\.PRICE_MAIN:\s*1[^\n]*\}",
        "role_priority = {NumericRole.CASHBACK: 4, NumericRole.CREDIT_PAYMENT: 4, NumericRole.VARIANT_PRICE: 2, NumericRole.PRICE_MAIN: 1}",
        text,
        count=1,
    )
    if count != 1 and "NumericRole.VARIANT_PRICE: 2" not in text:
        raise RuntimeError("role_priority introuvable")

    CLASSIFIER.write_text(text, encoding="utf-8")


def patch_resolver() -> None:
    text = RESOLVER.read_text(encoding="utf-8")

    if "def _variant_prices(" not in text:
        marker = "def _format_and_characteristics(\n"
        if marker not in text:
            raise RuntimeError("_format_and_characteristics introuvable")
        text = text.replace(marker, VARIANT_FORMATTER + marker, 1)

    text = replace_function(text, "_format_and_characteristics", FORMAT_FUNC, "_assemble")

    if "variant_prices = _variant_prices(facts)" not in text:
        marker = "        model = _extract_model(objects)\n"
        if marker not in text:
            raise RuntimeError("model = _extract_model(objects) introuvable")
        text = text.replace(
            marker,
            marker + "        variant_prices = _variant_prices(facts)\n",
            1,
        )

    if "variant=variant_prices" not in text:
        text, count = re.subn(
            r'brand=next\(\(brand for brand in brands if brand\), ""\), model=model,\s*\n\s*quantity=quantity,',
            'brand=next((brand for brand in brands if brand), ""), model=model, variant=variant_prices,\n            quantity=quantity,',
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("mapping Offer.variant introuvable")

    RESOLVER.write_text(text, encoding="utf-8")


def patch_product_model() -> None:
    text = MODELS.read_text(encoding="utf-8")
    if "variantes_prix: str" not in text:
        text, count = re.subn(
            r'(?m)^(\s*)caracteristiques:\s*str\s*=\s*""\s*$',
            lambda m: m.group(0) + f'\n{m.group(1)}variantes_prix: str = ""',
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("Product.caracteristiques introuvable")
    MODELS.write_text(text, encoding="utf-8")


def patch_pipeline() -> None:
    text = PIPELINE.read_text(encoding="utf-8")
    if "variantes_prix=offer.variant" not in text:
        old = '        caracteristiques=" • ".join(offer.technical_specs),\n'
        if old not in text:
            raise RuntimeError("mapping caracteristiques introuvable dans pipeline.py")
        text = text.replace(
            old,
            '        caracteristiques=" • ".join(offer.technical_specs), variantes_prix=offer.variant,\n',
            1,
        )
    PIPELINE.write_text(text, encoding="utf-8")


def patch_ui() -> None:
    text = INDEX.read_text(encoding="utf-8")
    if "Variantes / autres prix" not in text:
        old = "<th>Caractéristiques</th><th>Prix promo</th>"
        if old not in text:
            raise RuntimeError("colonnes HTML introuvables")
        text = text.replace(
            old,
            "<th>Caractéristiques</th><th>Variantes / autres prix</th><th>Prix promo</th>",
            1,
        )
    INDEX.write_text(text, encoding="utf-8")

    text = APPJS.read_text(encoding="utf-8")
    if "'variantes_prix'" not in text:
        old = "'caracteristiques','prix_promo'"
        if old not in text:
            raise RuntimeError("liste des colonnes app.js introuvable")
        text = text.replace(
            old,
            "'caracteristiques','variantes_prix','prix_promo'",
            1,
        )
    APPJS.write_text(text, encoding="utf-8")


def patch_exporter() -> None:
    text = EXPORTER.read_text(encoding="utf-8")

    if '("variantes_prix", "Variantes / autres prix")' not in text:
        old = '("caracteristiques", "Caractéristiques"), ("prix_promo", "Prix promo"),'
        if old not in text:
            raise RuntimeError("COLUMNS exporter introuvable")
        text = text.replace(
            old,
            '("caracteristiques", "Caractéristiques"),\n    ("variantes_prix", "Variantes / autres prix"), ("prix_promo", "Prix promo"),',
            1,
        )

    if '"Variantes / autres prix": 42' not in text:
        old = '"Format / conditionnement": 24, "Caractéristiques": 48,'
        if old not in text:
            raise RuntimeError("widths exporter introuvable")
        text = text.replace(
            old,
            '"Format / conditionnement": 24, "Caractéristiques": 48,\n        "Variantes / autres prix": 42,',
            1,
        )

    EXPORTER.write_text(text, encoding="utf-8")


def main() -> None:
    for path in [DOMAIN, CLASSIFIER, RESOLVER, MODELS, PIPELINE, INDEX, APPJS, EXPORTER]:
        if not path.exists():
            raise RuntimeError(f"fichier introuvable: {path}")

    ensure_domain()
    patch_classifier()
    patch_resolver()
    patch_product_model()
    patch_pipeline()
    patch_ui()
    patch_exporter()

    print("VARIANT PRICE FIX V3 APPLIQUE")
    print("Aucun fichier .bak cree.")
    print(r"Lance maintenant: .\test.ps1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        raise

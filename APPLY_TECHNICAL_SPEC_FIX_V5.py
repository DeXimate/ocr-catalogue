from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
CLASSIFIER = ROOT / "ocr_catalogue" / "semantics" / "classifier.py"
RESOLVER = ROOT / "ocr_catalogue" / "offers" / "resolver.py"
TESTS = ROOT / "tests" / "test_offer_engine.py"

TECHNICAL_HELPERS = '_TECHNICAL_MEASURE_UNIT = r"(?:BTU|WATTS?|W|TOURS?(?:/MIN)?|HZ|V|USB|CM|MM|POUCES?)"\n\n\ndef _normalize_technical_measure_text(text: str) -> str:\n    """Repair thousands split around a technical unit by PDF text ordering."""\n    clean = re.sub(r"\\s+", " ", (text or "")).strip()\n\n    # InDesign/PDF reading order can produce: "12 BTU ,000".\n    clean = re.sub(\n        rf"\\b(\\d{{1,3}})\\s*({_TECHNICAL_MEASURE_UNIT})\\s*[,.\\s]\\s*(\\d{{3}})\\b",\n        lambda match: f"{match.group(1)}{match.group(3)} {match.group(2)}",\n        clean,\n        flags=re.I,\n    )\n\n    # And sometimes: "12 ,000 BTU" or "12 000 BTU".\n    clean = re.sub(\n        rf"\\b(\\d{{1,3}})\\s*[,.\\s]\\s*(\\d{{3}})\\s*({_TECHNICAL_MEASURE_UNIT})\\b",\n        lambda match: f"{match.group(1)}{match.group(2)} {match.group(3)}",\n        clean,\n        flags=re.I,\n    )\n    return clean\n\n\ndef _technical_measure_atoms(text: str) -> list[str]:\n    """Extract atomic technical measures instead of exporting a whole table row."""\n    clean = _normalize_technical_measure_text(text)\n    pattern = re.compile(\n        rf"\\b\\d{{1,6}}(?:[,.]\\d+)?\\s*{_TECHNICAL_MEASURE_UNIT}\\b",\n        re.I,\n    )\n    return list(dict.fromkeys(re.sub(r"\\s+", " ", match.group(0)).strip() for match in pattern.finditer(clean)))\n\n\n'
CLASSIFY_NON_PRICE = 'def _classify_non_price_numbers(page: PageScene) -> list[NumericFact]:\n    facts = []\n    words = [item for item in page.objects if item.raw_type == "word"]\n\n    for marker in [word for word in words if word.text.strip() == "%"]:\n        heads = [\n            word for word in words\n            if re.fullmatch(r"\\d{1,2}", word.text.strip())\n            and -word.font_size * .08 <= marker.bbox.x0 - word.bbox.x1 <= word.font_size * .65\n            and abs(marker.bbox.cy - word.bbox.cy) <= max(marker.font_size, word.font_size) * .6\n        ]\n        if heads:\n            head = min(\n                heads,\n                key=lambda word: abs(marker.bbox.x0 - word.bbox.x1) + abs(marker.bbox.cy - word.bbox.cy),\n            )\n            facts.append(NumericFact(\n                f"{head.id}-{marker.id}-discount",\n                page.number,\n                f"{head.text}%",\n                head.text,\n                head.bbox.union(marker.bbox),\n                NumericRole.DISCOUNT,\n                .98,\n                [head.id, marker.id],\n                ["badge_pourcentage"],\n            ))\n\n    for obj in [item for item in page.objects if item.raw_type == "line"]:\n        raw_text = obj.text.strip()\n        text = _normalize_technical_measure_text(raw_text)\n\n        if match := PERCENT.search(text):\n            facts.append(NumericFact(\n                f"{obj.id}-discount",\n                page.number,\n                match.group(0),\n                match.group(1),\n                obj.bbox,\n                NumericRole.DISCOUNT,\n                .96,\n                obj.source_ids,\n                ["symbole_pourcentage"],\n            ))\n\n        # Extract technical numbers atomically. A row such as\n        # "Puissance Froid Chaud Prix 9000 BTU 12000 BTU" yields two facts,\n        # not one polluted characteristic containing the table headers.\n        atoms = _technical_measure_atoms(text)\n        for index, atom in enumerate(atoms):\n            role = NumericRole.POWER if re.search(r"\\b(?:watts?|w)\\b", atom, re.I) else NumericRole.TECHNICAL_SPEC\n            facts.append(NumericFact(\n                f"{obj.id}-{role.value.lower()}-{index}",\n                page.number,\n                atom,\n                atom,\n                obj.bbox,\n                role,\n                .94,\n                obj.source_ids,\n                ["mesure_technique_atomique"],\n            ))\n\n        if PRICE_BASIS.search(text):\n            facts.append(NumericFact(\n                f"{obj.id}-{NumericRole.PRICE_BASIS.value.lower()}",\n                page.number,\n                text,\n                text,\n                obj.bbox,\n                NumericRole.PRICE_BASIS,\n                .9,\n                obj.source_ids,\n                ["grammaire_contextuelle"],\n            ))\n            continue\n\n        # Duration is extracted independently, so it can coexist with BTU/W.\n        duration_match = re.search(r"\\bgarantie\\s+\\d+\\s*(?:ans?|mois)\\b", text, re.I)\n        if duration_match:\n            value = re.sub(r"\\s+", " ", duration_match.group(0)).strip()\n            facts.append(NumericFact(\n                f"{obj.id}-{NumericRole.DURATION.value.lower()}",\n                page.number,\n                value,\n                value,\n                obj.bbox,\n                NumericRole.DURATION,\n                .92,\n                obj.source_ids,\n                ["garantie_expresse"],\n            ))\n        elif re.fullmatch(r"\\s*\\d+\\s*(?:ans?|mois)\\s*", text, re.I):\n            facts.append(NumericFact(\n                f"{obj.id}-{NumericRole.DURATION.value.lower()}",\n                page.number,\n                text,\n                text,\n                obj.bbox,\n                NumericRole.DURATION,\n                .86,\n                obj.source_ids,\n                ["duree_contextuelle"],\n            ))\n\n        dimension_match = re.search(r"\\b\\d+(?:[,.]\\d+)?\\s*[x×]\\s*\\d+(?:[,.]\\d+)?\\s*(?:cm|mm)\\b", text, re.I)\n        if dimension_match:\n            value = re.sub(r"\\s+", " ", dimension_match.group(0)).strip()\n            facts.append(NumericFact(\n                f"{obj.id}-{NumericRole.DIMENSION.value.lower()}",\n                page.number,\n                value,\n                value,\n                obj.bbox,\n                NumericRole.DIMENSION,\n                .92,\n                obj.source_ids,\n                ["dimension_atomique"],\n            ))\n\n        # Keep legacy quantity/model behaviour only when no atomic technical\n        # measure already explains the line.\n        if not atoms and QUANTITY.search(text):\n            role = NumericRole.PACK_SIZE if re.search(r"lot|[x×]", text, re.I) else NumericRole.QUANTITY\n            facts.append(NumericFact(\n                f"{obj.id}-{role.value.lower()}",\n                page.number,\n                text,\n                text,\n                obj.bbox,\n                role,\n                .9,\n                obj.source_ids,\n                ["grammaire_contextuelle"],\n            ))\n        elif (\n            not atoms\n            and MODEL.search(text)\n            and not re.fullmatch(r"\\s*\\d{1,4}\\s*DT\\s*", text, re.I)\n        ):\n            facts.append(NumericFact(\n                f"{obj.id}-{NumericRole.MODEL.value.lower()}",\n                page.number,\n                text,\n                text,\n                obj.bbox,\n                NumericRole.MODEL,\n                .72,\n                obj.source_ids,\n                ["grammaire_contextuelle"],\n            ))\n\n    unique: list[NumericFact] = []\n    for fact in facts:\n        duplicate = next(\n            (\n                other for other in unique\n                if other.role == fact.role\n                and other.value == fact.value\n                and other.bbox.distance(fact.bbox) < 4\n            ),\n            None,\n        )\n        if duplicate is None:\n            unique.append(fact)\n    return unique\n\n\n'
RESOLVER_HELPERS = '_TECHNICAL_HEADER_WORDS = {"puissance", "froid", "chaud", "prix"}\n_MONEYISH_MODEL = re.compile(r"^\\s*\\+?\\d{1,4}(?:[,.]\\d{3})?\\s*DT\\s*$", re.I)\n_MODEL_TOKEN = re.compile(r"\\b[A-Z0-9][A-Z0-9._/-]{4,}\\b")\n\n\ndef _normalize_technical_text(text: str) -> str:\n    """Normalize display text for technical characteristics."""\n    clean = re.sub(r"\\s+", " ", (text or "")).strip(" -•")\n\n    unit = r"(?:BTU|WATTS?|W|TOURS?(?:/MIN)?|HZ|V|USB|CM|MM|POUCES?)"\n    clean = re.sub(\n        rf"\\b(\\d{{1,3}})\\s*({unit})\\s*[,.\\s]\\s*(\\d{{3}})\\b",\n        lambda match: f"{match.group(1)}{match.group(3)} {match.group(2)}",\n        clean,\n        flags=re.I,\n    )\n    clean = re.sub(\n        rf"\\b(\\d{{1,3}})\\s*[,.\\s]\\s*(\\d{{3}})\\s*({unit})\\b",\n        lambda match: f"{match.group(1)}{match.group(2)} {match.group(3)}",\n        clean,\n        flags=re.I,\n    )\n    return clean\n\n\ndef _is_structural_technical_text(text: str) -> bool:\n    """Reject mini-table headers that describe layout, not the product."""\n    clean = _normalize_technical_text(text)\n    lowered = clean.lower().strip(" :;-")\n\n    if lowered == "prix":\n        return True\n\n    # Keep the legitimate feature "CHAUD/FROID".\n    if re.fullmatch(r"chaud\\s*/\\s*froid", lowered, re.I):\n        return False\n\n    words = re.findall(r"[a-zà-ÿ]+", lowered)\n    if len(words) >= 2 and set(words).issubset(_TECHNICAL_HEADER_WORDS):\n        return True\n\n    if "prix" in words and any(word in {"puissance", "froid", "chaud"} for word in words):\n        return True\n\n    return False\n\n\ndef _model_token_from_text(text: str) -> str:\n    """Find a plausible appliance model while rejecting prices such as 148DT."""\n    candidates: list[str] = []\n    for match in _MODEL_TOKEN.finditer(text or ""):\n        token = match.group(0).strip("-./")\n        if not token or _MONEYISH_MODEL.fullmatch(token):\n            continue\n        if re.fullmatch(r"\\d+(?:DT|DTT)", token, re.I):\n            continue\n        if not (re.search(r"[A-Z]", token) and re.search(r"\\d", token)):\n            continue\n        # Avoid short technical codes such as R32; models are normally longer\n        # references and/or carry separators.\n        if len(token) < 6 and not re.search(r"[-_/]", token):\n            continue\n        candidates.append(token)\n\n    if not candidates:\n        return ""\n    return max(candidates, key=lambda token: (len(token), token.count("-") + token.count("/")))\n\n\n'
EXTRACT_MODEL = 'def _extract_model(objects: list[VisualObject]) -> str:\n    explicit = [\n        obj.text.strip()\n        for obj in objects\n        if obj.semantic_role == SemanticRole.MODEL\n        and obj.text.strip()\n        and not _MONEYISH_MODEL.fullmatch(obj.text.strip())\n        and not re.fullmatch(r"\\d+(?:DT|DTT)", obj.text.strip(), re.I)\n    ]\n    if explicit:\n        return explicit[0]\n\n    # Brand and model can share the same physical line.\n    for obj in objects:\n        if obj.semantic_role != SemanticRole.BRAND:\n            continue\n        match = _MODEL_AFTER_BRAND.search(obj.text)\n        if not match:\n            continue\n        token = match.group(1).strip("-./")\n        if (\n            not _MONEYISH_MODEL.fullmatch(token)\n            and re.search(r"[A-Z]", token)\n            and re.search(r"\\d", token)\n        ):\n            return token\n\n    # Fallback for lines such as:\n    # MX-CH12T-INV4-S - T3\n    # GWH18AWDXB-K6DNA1B - R32\n    candidates = [\n        _model_token_from_text(obj.text)\n        for obj in objects\n        if obj.text.strip()\n    ]\n    candidates = [candidate for candidate in candidates if candidate]\n    return max(candidates, key=len) if candidates else ""\n\n\n'
FORMAT_CHARACTERISTICS = 'def _format_and_characteristics(\n    product: str,\n    objects: list[VisualObject],\n    facts: list[NumericFact],\n) -> tuple[str, list[str]]:\n    quantities = [\n        _normalize_technical_text(obj.text)\n        for obj in objects\n        if obj.semantic_role == SemanticRole.QUANTITY\n    ]\n    appliance = _is_appliance_offer(product, objects)\n\n    characteristics: list[str] = []\n\n    for obj in objects:\n        if obj.semantic_role != SemanticRole.TECHNICAL_SPEC:\n            continue\n\n        text = _normalize_technical_text(obj.text)\n        if not text or _CREDIT_ONLY.search(text):\n            continue\n        if re.search(r"\\b[àa]\\s*$", text, re.I):\n            # Reference line for a variant price, e.g. "Existe en 180 cm à".\n            continue\n        if _is_structural_technical_text(text):\n            continue\n\n        # On appliance mini-tables, the line can contain headers plus several\n        # numeric facts. Those facts are exported atomically below; do not\n        # duplicate the polluted whole row here.\n        if (\n            re.search(r"\\bprix\\b", text, re.I)\n            and re.search(r"\\b(?:puissance|froid|chaud)\\b", text, re.I)\n        ):\n            continue\n\n        if appliance or _CHARACTERISTIC_CUE.search(text):\n            characteristics.append(text)\n\n    # Numeric facts are the authoritative source for technical measurements.\n    for fact in facts:\n        if fact.role == NumericRole.VARIANT_PRICE or "prix_variante_non_exporte" in fact.evidence:\n            continue\n        if fact.role not in {\n            NumericRole.POWER,\n            NumericRole.CAPACITY,\n            NumericRole.DIMENSION,\n            NumericRole.DURATION,\n            NumericRole.TECHNICAL_SPEC,\n        }:\n            continue\n\n        text = _normalize_technical_text(fact.text)\n        if not text or _CREDIT_ONLY.search(text) or _is_structural_technical_text(text):\n            continue\n        if fact.role == NumericRole.DURATION and not re.search(r"garantie", text, re.I):\n            continue\n        characteristics.append(text)\n\n    if appliance:\n        characteristics.extend(\n            quantity for quantity in quantities\n            if quantity and not _is_structural_technical_text(quantity)\n        )\n        retail_format = ""\n    else:\n        retail_format = quantities[-1] if quantities else ""\n\n    return retail_format, _dedupe_text(characteristics)\n\n\n'
TESTS_BLOCK = '    def test_split_thousands_btu_is_normalized(self):\n        self.assertEqual(_normalize_technical_text("12 BTU ,000"), "12000 BTU")\n        self.assertEqual(_normalize_technical_text("18 ,000 BTU"), "18000 BTU")\n\n    def test_appliance_table_headers_are_not_characteristics(self):\n        objects = [\n            VisualObject(\n                "header", 1, "line", BBox(0, 0, 120, 10),\n                text="Puissance Froid Chaud Prix",\n                semantic_role=SemanticRole.TECHNICAL_SPEC,\n                semantic_confidence=.9,\n            ),\n            VisualObject(\n                "feature", 1, "line", BBox(0, 20, 80, 30),\n                text="CHAUD/FROID",\n                semantic_role=SemanticRole.TECHNICAL_SPEC,\n                semantic_confidence=.9,\n            ),\n        ]\n        facts = [\n            NumericFact(\n                "btu", 1, "12 BTU ,000", "12000 BTU",\n                BBox(0, 40, 80, 50),\n                NumericRole.TECHNICAL_SPEC, .94,\n                evidence=["mesure_technique_atomique"],\n            ),\n        ]\n        retail_format, characteristics = _format_and_characteristics("Climatiseur", objects, facts)\n        self.assertEqual(retail_format, "")\n        self.assertIn("CHAUD/FROID", characteristics)\n        self.assertIn("12000 BTU", characteristics)\n        self.assertFalse(any("Prix" in value for value in characteristics))\n        self.assertFalse(any("Puissance Froid Chaud" in value for value in characteristics))\n\n    def test_money_amount_cannot_be_model(self):\n        objects = [\n            VisualObject(\n                "bad-model", 1, "line", BBox(0, 0, 30, 10),\n                text="148DT",\n                semantic_role=SemanticRole.MODEL,\n                semantic_confidence=.72,\n            ),\n            VisualObject(\n                "real-model", 1, "line", BBox(0, 20, 120, 30),\n                text="GWH18AWDXB-K6DNA1B - R32",\n                semantic_role=SemanticRole.TECHNICAL_SPEC,\n                semantic_confidence=.9,\n            ),\n        ]\n        self.assertEqual(_extract_model(objects), "GWH18AWDXB-K6DNA1B")\n\n'


def replace_function(text: str, name: str, replacement: str, next_name: str) -> str:
    pattern = rf"def {re.escape(name)}\(.*?(?=\ndef {re.escape(next_name)}\()"
    updated, count = re.subn(
        pattern,
        replacement.rstrip() + "\n\n",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"fonction {name} introuvable")
    return updated


def patch_classifier() -> None:
    text = CLASSIFIER.read_text(encoding="utf-8")

    if "_normalize_technical_measure_text" not in text:
        marker = "def _classify_non_price_numbers(page: PageScene) -> list[NumericFact]:\n"
        if marker not in text:
            raise RuntimeError("_classify_non_price_numbers introuvable")
        text = text.replace(marker, TECHNICAL_HELPERS + marker, 1)

    text = replace_function(
        text,
        "_classify_non_price_numbers",
        CLASSIFY_NON_PRICE,
        "_is_actual_promotion_line",
    )

    compile(text, str(CLASSIFIER), "exec")
    CLASSIFIER.write_text(text, encoding="utf-8")


def patch_resolver() -> None:
    text = RESOLVER.read_text(encoding="utf-8")

    if "_normalize_technical_text" not in text:
        marker = "def _is_appliance_offer(product: str, objects: list[VisualObject]) -> bool:\n"
        if marker not in text:
            raise RuntimeError("_is_appliance_offer introuvable")
        text = text.replace(marker, RESOLVER_HELPERS + marker, 1)

    text = replace_function(text, "_extract_model", EXTRACT_MODEL, "_dedupe_text")
    text = replace_function(
        text,
        "_format_and_characteristics",
        FORMAT_CHARACTERISTICS,
        "_assemble",
    )

    compile(text, str(RESOLVER), "exec")
    RESOLVER.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")

    # Extend the existing resolver import with the new test helpers.
    match = re.search(r"^from ocr_catalogue\.offers\.resolver import .+$", text, flags=re.M)
    if not match:
        raise RuntimeError("import resolver introuvable dans tests")
    line = match.group(0)
    for helper in ["_normalize_technical_text", "_is_structural_technical_text"]:
        if helper not in line:
            line += f", {helper}"
    text = text[:match.start()] + line + text[match.end():]

    if "def test_split_thousands_btu_is_normalized" not in text:
        marker = "    def test_classifier_accepts_only_explicit_free_mechanism(self):\n"
        if marker not in text:
            raise RuntimeError("point d'insertion des tests introuvable")
        text = text.replace(marker, TESTS_BLOCK + marker, 1)

    compile(text, str(TESTS), "exec")
    TESTS.write_text(text, encoding="utf-8")


def main() -> None:
    for path in (CLASSIFIER, RESOLVER, TESTS):
        if not path.exists():
            raise RuntimeError(f"fichier introuvable: {path}")

    patch_classifier()
    patch_resolver()
    patch_tests()

    print("TECHNICAL SPEC FIX V5 APPLIQUE")
    print("Corrections:")
    print(" - 12 BTU ,000 -> 12000 BTU")
    print(" - suppression des en-tetes Puissance / Froid / Chaud / Prix")
    print(" - extraction atomique des mesures techniques")
    print(" - 148DT ne peut plus etre un modele")
    print(r"Etape suivante: .\test.ps1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        raise

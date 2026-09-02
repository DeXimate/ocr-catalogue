from __future__ import annotations

import re
import statistics
from collections import Counter

from ..domain import BBox, DocumentScene, NumericFact, NumericRole, PageScene, SemanticRole, VisualObject


ARABIC = re.compile(r"[\u0600-\u06ff]")
PRICE_COMPACT = re.compile(r"^(\d{1,4})\s*[,\.]\s*(\d{3})\s*(?:DT)?$", re.I)
PERCENT = re.compile(r"(?<!\d)(\d{1,2})\s*%")
PRICE_BASIS = re.compile(r"\b(?:LE\s+KG|LES\s+\d+\s*(?:G|KG|ML|CL|L)|LA\s+PI[EÈ]CE)\b", re.I)
QUANTITY = re.compile(
    r"\b(?:lot\s+de\s+)?\d+(?:[,.]\d+)?(?:\s*[x×]\s*\d+(?:[,.]\d+)?)?\s*"
    r"(?:mg|g|kg|ml|cl|l|litres?|pi[eè]ces?|rouleaux|doses?|portions?|tranches?|plis?|pages?|programmes?)\b|"
    r"\b(?:le\s+blister\s+de|les)\s+\d+\b", re.I,
)
TECHNICAL = re.compile(r"\b\d+(?:[,.]\d+)?\s*(?:watts?|w|btu|tours?|hz|v|usb|cm|mm|pouces?)\b|\bgarantie\s+\d+\s*(?:ans?|mois)\b", re.I)
MODEL = re.compile(r"\b(?=[A-Z0-9-]{3,}\b)(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b")
FREE_MECHANISM = re.compile(r"\b(?:GRATUIT(?:E|ES|S)?|OFFERT(?:E|ES|S)?)\b", re.I)
CASHBACK_MECHANISM = re.compile(r"\b(?:CASHBACK|VERS[ÉE]S?)\b", re.I)
OFFER_RATIO = re.compile(r"\b\d+\s*\+\s*\d+\b")
SECOND_ITEM = re.compile(r"\bSUR\s+LE\s+\d(?:ER|E|ÈME|EME)?\b", re.I)
TECHNICAL_COMPOSITION = re.compile(
    r"(?:\d+(?:[,.]\d+)?\s*[x×]\s*){2,}\d+(?:[,.]\d+)?\s*(?:cm|mm)\b|"
    r"\b\d+(?:[,.]\d+)?(?:\s*\+\s*\d+(?:[,.]\d+)?)+\s*(?:litres?|l|ml|cl|pi[eè]ces?)\b|"
    r"^\s*\+\s*\d+\s+(?:faitouts?|casseroles?|po[eê]les?|couvercles?|accessoires?)\b",
    re.I,
)
NOISE_WORDS = re.compile(r"photos? non contractuelles?|dans la limite des stocks|monoprix|produits frais", re.I)


def _overprint_digits(value: str) -> str:
    groups = re.findall(r"((\d)\2*)", value)
    if groups and "".join(group for group, _ in groups) == value and all(len(group) % 2 == 0 for group, _ in groups):
        return "".join(digit * (len(group) // 2) for group, digit in groups)
    return value


def _nearby_lines(page: PageScene, bbox: BBox, radius: float) -> list[VisualObject]:
    return [obj for obj in page.objects if obj.raw_type == "line" and obj.bbox.distance(bbox) <= radius]


def _numeric_context(page: PageScene, bbox: BBox) -> str:
    radius = max(14.0, statistics.median([obj.font_size for obj in page.objects if obj.raw_type == "word"] or [8.0]) * 3.5)
    return " ".join(obj.text for obj in _nearby_lines(page, bbox, radius))


def _preceded_by_reference_cue(page: PageScene, bbox: BBox) -> bool:
    words = [obj for obj in page.objects if obj.raw_type == "word"]
    if any(obj.text.lower().strip(" :") in {"à", "avant"} and 0 <= bbox.x0 - obj.bbox.x1 <= max(30, obj.font_size * 4) and abs(bbox.cy - obj.bbox.cy) <= max(20, obj.font_size * 2) for obj in words):
        return True
    return False


def _price_role(page: PageScene, bbox: BBox) -> tuple[NumericRole, float, list[str]]:
    context = _numeric_context(page, bbox)
    plus_near = any(
        obj.raw_type == "word" and obj.text.strip().startswith("+")
        and obj.bbox.distance(bbox) <= max(12.0, bbox.height * 1.8)
        for obj in page.objects
    )
    if plus_near and re.search(r"VERS[ÉE]S?", context, re.I):
        return NumericRole.CASHBACK, .96, ["voisinage_verses"]
    body_size = statistics.median([obj.font_size for obj in page.objects if obj.raw_type == "word" and obj.font_size > 0] or [8.0])
    if re.search(r"ACHAT\s+[ÀA]\s+CR[ÉE]DIT", context, re.I) and re.search(r"\bmois\b", context, re.I) and bbox.height <= body_size * 2.5:
        return NumericRole.CREDIT_PAYMENT, .92, ["voisinage_credit"]
    if _preceded_by_reference_cue(page, bbox):
        return NumericRole.TECHNICAL_SPEC, .9, ["prix_variante_non_exporte"]
    return NumericRole.PRICE_MAIN, .72, ["expression_dt_complete"]


def _find_prices(page: PageScene) -> list[NumericFact]:
    words = [obj for obj in page.objects if obj.raw_type == "word"]
    decimal_tails: list[tuple[str, BBox, list[str], float]] = []
    for word in words:
        if re.match(r"^[,.]\d{3}$", word.text):
            decimal_tails.append((word.text, word.bbox, [word.id], word.font_size))
        elif re.match(r"^\d{3}$", word.text):
            markers = [
                marker for marker in words
                if marker.text in {",", "."}
                and -max(2.0, word.font_size * .15) <= marker.bbox.x1 - word.bbox.x0 <= max(3.0, word.font_size * .25)
                and abs(marker.bbox.cy - word.bbox.cy) <= max(marker.font_size, word.font_size) * .35
            ]
            if markers:
                marker = min(markers, key=lambda item: abs(item.bbox.x1 - word.bbox.x0))
                decimal_tails.append((marker.text + word.text, marker.bbox.union(word.bbox), [marker.id, word.id], max(marker.font_size, word.font_size)))
    candidates: list[NumericFact] = []
    for head in words:
        clean = head.text.replace(" ", "").upper()
        compact = PRICE_COMPACT.match(clean)
        if compact and "DT" in clean:
            value = f"{int(compact.group(1))},{compact.group(2)}"
            role, confidence, evidence = _price_role(page, head.bbox)
            candidates.append(NumericFact(f"{head.id}-price", page.number, head.text, value, head.bbox, role, confidence, [head.id], evidence))
            continue
        embedded = re.match(r"^(\d{1,4})D+T+$", clean)
        number = re.match(r"^(\d{1,4})$", clean)
        if not embedded and not number:
            continue
        amount = (embedded or number).group(1)
        # A number immediately followed by % is a discount badge, never a
        # price head. This prevents 32 % from becoming 32,000 DT by borrowing
        # the currency of a neighbouring offer.
        if any(
            obj.text.strip() == "%"
            and -head.font_size * .08 <= obj.bbox.x0 - head.bbox.x1 <= head.font_size * .55
            and abs(obj.bbox.cy - head.bbox.cy) <= max(head.font_size, obj.font_size) * .55
            for obj in words
        ):
            continue
        currencies = [head] if embedded else [
            obj for obj in words
            if obj.text.upper().replace(" ", "") in {"DT", "DTT"}
            and -head.font_size * .12 <= obj.bbox.x0 - head.bbox.x1 <= head.font_size * .8
            and abs(obj.bbox.cy - head.bbox.cy) <= max(head.font_size, obj.font_size) * .75
        ]
        for currency in currencies[:1]:
            if not embedded and head.bbox.height < currency.bbox.height * 1.45:
                continue
            tails = [item for item in decimal_tails if -max(currency.font_size, head.font_size) * 1.25 <= item[1].x0 - currency.bbox.x1 <= max(currency.font_size, head.font_size) * 2.3 and abs(item[1].cy - currency.bbox.cy) <= max(currency.font_size, head.font_size) * 1.8]
            if not tails:
                if head.bbox.height >= max(18, currency.font_size * 2) and int(amount) >= 20:
                    bbox = head.bbox
                    role, confidence, evidence = _price_role(page, bbox)
                    candidates.append(NumericFact(f"{head.id}-price", page.number, head.text, f"{int(amount)},000", bbox, role, confidence - .18, [head.id], evidence + ["milliemes_implicites"] ))
                continue
            tail_text, tail_bbox, tail_ids, _ = min(tails, key=lambda item: abs(item[1].x0 - currency.bbox.x1) + abs(item[1].cy - currency.bbox.cy))
            bbox = head.bbox.union(currency.bbox).union(tail_bbox)
            value = f"{int(amount)},{tail_text[1:]}"
            role, confidence, evidence = _price_role(page, bbox)
            candidates.append(NumericFact(f"{head.id}-{'-'.join(tail_ids)}-price", page.number, f"{amount} DT {tail_text}", value, bbox, role, confidence, [head.id, currency.id, *tail_ids], evidence))
    unique: list[NumericFact] = []
    role_priority = {NumericRole.CASHBACK: 3, NumericRole.CREDIT_PAYMENT: 3, NumericRole.PRICE_MAIN: 1}
    for fact in sorted(candidates, key=lambda item: (-role_priority.get(item.role, 0), -item.confidence, item.bbox.top)):
        duplicate = next((other for other in unique if other.value == fact.value and other.bbox.distance(fact.bbox) < 3), None)
        if duplicate is None:
            unique.append(fact)
    return unique


def _classify_non_price_numbers(page: PageScene) -> list[NumericFact]:
    facts = []
    words = [item for item in page.objects if item.raw_type == "word"]
    for marker in [word for word in words if word.text.strip() == "%"]:
        heads = [
            word for word in words
            if re.fullmatch(r"\d{1,2}", word.text.strip())
            and -word.font_size * .08 <= marker.bbox.x0 - word.bbox.x1 <= word.font_size * .65
            and abs(marker.bbox.cy - word.bbox.cy) <= max(marker.font_size, word.font_size) * .6
        ]
        if heads:
            head = min(heads, key=lambda word: abs(marker.bbox.x0 - word.bbox.x1) + abs(marker.bbox.cy - word.bbox.cy))
            facts.append(NumericFact(
                f"{head.id}-{marker.id}-discount", page.number, f"{head.text}%", head.text,
                head.bbox.union(marker.bbox), NumericRole.DISCOUNT, .98,
                [head.id, marker.id], ["badge_pourcentage"],
            ))
    for obj in [item for item in page.objects if item.raw_type == "line"]:
        text = obj.text.strip()
        role = None
        confidence = .9
        if match := PERCENT.search(text):
            facts.append(NumericFact(f"{obj.id}-discount", page.number, match.group(0), match.group(1), obj.bbox, NumericRole.DISCOUNT, .96, obj.source_ids, ["symbole_pourcentage"]))
        if PRICE_BASIS.search(text):
            role = NumericRole.PRICE_BASIS
        elif re.search(r"\b(?:watts?|btu|tours?|hz|usb)\b", text, re.I):
            role = NumericRole.POWER if re.search(r"watts?|\bw\b", text, re.I) else NumericRole.TECHNICAL_SPEC
        elif re.search(r"\bgarantie\b|\b\d+\s*(?:ans?|mois)\b", text, re.I):
            role = NumericRole.DURATION
        elif re.search(r"\b\d+\s*[x×]\s*\d+\s*(?:cm|mm)\b", text, re.I):
            role = NumericRole.DIMENSION
        elif QUANTITY.search(text):
            role = NumericRole.PACK_SIZE if re.search(r"lot|[x×]", text, re.I) else NumericRole.QUANTITY
        elif MODEL.search(text):
            role = NumericRole.MODEL
            confidence = .72
        if role:
            facts.append(NumericFact(f"{obj.id}-{role.value.lower()}", page.number, text, text, obj.bbox, role, confidence, obj.source_ids, ["grammaire_contextuelle"] ))
    unique = []
    for fact in facts:
        duplicate = next((other for other in unique if other.role == fact.role and other.value == fact.value and other.bbox.distance(fact.bbox) < 4), None)
        if duplicate is None:
            unique.append(fact)
    return unique


def _is_actual_promotion_line(page: PageScene, obj: VisualObject) -> bool:
    text = obj.text.replace("ERFFO", "OFFRE").strip()
    if TECHNICAL_COMPOSITION.search(text):
        return False
    if CASHBACK_MECHANISM.search(text) or SECOND_ITEM.search(text):
        return True
    if FREE_MECHANISM.search(text):
        return True
    if OFFER_RATIO.search(text):
        radius = max(18.0, obj.font_size * 4.0)
        return any(
            other.id != obj.id and FREE_MECHANISM.search(other.text)
            and other.bbox.distance(obj.bbox) <= radius
            for other in page.objects if other.raw_type == "line"
        )
    return False


def _classify_lines(page: PageScene) -> None:
    lines = [obj for obj in page.objects if obj.raw_type == "line"]
    body_sizes = [obj.font_size for obj in lines if re.search(r"[A-Za-zÀ-ÿ\u0600-\u06ff]", obj.text)]
    median_size = statistics.median(body_sizes) if body_sizes else 8.0
    for obj in lines:
        text = obj.text.strip()
        is_bold = bool(re.search(r"(?:bold|black|heavy|semi[- ]?bold|demi)", obj.font_name, re.I))
        if not text:
            continue
        if NOISE_WORDS.search(text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.HEADER_FOOTER, .7
        elif ARABIC.search(text) and not re.search(r"[A-Za-zÀ-ÿ]", text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.ARABIC_TEXT, .93
        elif PRICE_BASIS.search(text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.PRICE_BASIS, .96
        elif re.search(r"d.?économie|^\s*DT\b|[,\.]\d{3}\b", text, re.I):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.RAW_TEXT, .85
        elif TECHNICAL_COMPOSITION.search(text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.TECHNICAL_SPEC, .94
        elif _is_actual_promotion_line(page, obj):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.PROMOTION, .9
        elif re.search(r"[“\"]([^”\"]+)[”\"]", text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.BRAND, .94
        elif TECHNICAL.search(text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.TECHNICAL_SPEC, .9
        elif QUANTITY.search(text):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.QUANTITY, .9
        elif MODEL.search(text) and not re.search(r"\s", text.strip()):
            obj.semantic_role, obj.semantic_confidence = SemanticRole.MODEL, .72
        elif len(text) >= 3 and re.search(r"[A-Za-zÀ-ÿ]", text) and not PERCENT.search(text) and not re.search(r"\bDT\b", text, re.I):
            if is_bold:
                obj.semantic_role = SemanticRole.PRODUCT_TEXT
                length_score = min(1.0, len(text) / 18)
                obj.semantic_confidence = .62 + .18 * length_score + (.08 if obj.font_size >= median_size else 0)
            else:
                # Monoprix uses the regular face for variants, flavours,
                # colours and explanatory copy. Keep it as offer context but
                # never let it seed or name an OFFER.
                obj.semantic_role = SemanticRole.TECHNICAL_SPEC
                obj.semantic_confidence = .78


def parse_promotion(objects: list[VisualObject], cashback: str = "") -> str:
    if cashback:
        return cashback
    lines = [obj.text.replace("ERFFO", "OFFRE").strip() for obj in sorted(objects, key=lambda item: (item.bbox.cy, item.bbox.x0)) if obj.semantic_role == SemanticRole.PROMOTION]
    joined = " ".join(lines)
    if not joined:
        return ""
    if TECHNICAL_COMPOSITION.search(joined) and not (FREE_MECHANISM.search(joined) or CASHBACK_MECHANISM.search(joined) or SECOND_ITEM.search(joined)):
        return ""
    plus_item = re.search(r"\+\s*(\d+\s+)?([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]{0,50}?)\s+(GRATUIT(?:E|ES|S)?|OFFERT(?:E|ES|S)?)\b", joined, re.I)
    if plus_item:
        count = plus_item.group(1) or ""
        item = re.sub(r"\s+", " ", plus_item.group(2)).strip()
        suffix = plus_item.group(3).lower()
        return f"+ {count}{item} {suffix}".strip()
    plus = re.search(r"(\d+)\s*\+\s*(\d+)", joined)
    if plus and FREE_MECHANISM.search(joined):
        return f"{int(plus.group(1))} achetés + {int(plus.group(2))} gratuit"
    dont = re.search(r"DONT\s+(\d+)\s*(LITRES?|PI[EÈ]CES?|UNIT[EÉ]S?)?.*?(GRATUIT(?:E|ES|S)?|OFFERT(?:E|ES|S)?)", joined, re.I)
    if dont:
        return f"{int(dont.group(1))} {(dont.group(2) or 'produit').lower()} gratuit"
    if SECOND_ITEM.search(joined):
        return " ".join(dict.fromkeys(line for line in lines if SECOND_ITEM.search(line) or re.search(r"%|r[ée]duction", line, re.I)))
    if FREE_MECHANISM.search(joined):
        return " ".join(dict.fromkeys(line for line in lines if FREE_MECHANISM.search(line)))
    return ""


def classify_document(document: DocumentScene) -> DocumentScene:
    for page in document.pages:
        _classify_lines(page)
        page.numeric_facts = _find_prices(page) + _classify_non_price_numbers(page)
    return document

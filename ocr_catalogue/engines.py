from __future__ import annotations

import re
import subprocess
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, ImageChops, ImageFilter, ImageOps

from .models import Product


PRICE_HEAD = re.compile(r"^(\d{1,4})\s*[,\.]\s*(\d{3})(?:DT)?$", re.I)
PERCENT = re.compile(r"^(\d{1,2})\s*%$")
ARABIC = re.compile(r"[\u0600-\u06ff]")
QUANTITY = re.compile(r"\b(?:\d+(?:[,.]\d+)?\s*(?:g|kg|ml|cl|l|litres?|pi[eè]ces?|rouleaux)|les\s+\d+|lot\s+de\s+\d+)\b", re.I)
NOISE = re.compile(r"(?:MONOPRIX|PRODUITS|articles sont disponibles|photos non contractuelles|d.?économie)", re.I)


@dataclass
class Token:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


def _dedupe_overprint_digits(value: str) -> str:
    groups = re.findall(r"((\d)\2*)", value)
    if groups and "".join(group for group, _ in groups) == value and all(len(group) % 2 == 0 for group, _ in groups):
        return "".join(digit * (len(group) // 2) for group, digit in groups)
    return value


def _poppler_binary(name: str) -> Path | None:
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin" / f"{name}.exe"
    return bundled if bundled.exists() else None


def render_pdf(source: Path, pages_dir: Path, dpi: int = 150) -> list[Path]:
    tool = _poppler_binary("pdftoppm")
    if not tool:
        raise RuntimeError("Poppler/pdftoppm est requis pour rendre les pages PDF")
    prefix = pages_dir / "page"
    subprocess.run([str(tool), "-jpeg", "-r", str(dpi), str(source), str(prefix)], check=True, capture_output=True)
    return sorted(pages_dir.glob("page-*.jpg"))


def _merge_price_tokens(tokens: list[Token]) -> list[tuple[str, Token]]:
    found: list[tuple[str, Token]] = []
    for token in tokens:
        clean = token.text.replace(" ", "").upper()
        match = PRICE_HEAD.match(clean)
        if match and "DT" in clean:
            found.append((f"{int(match.group(1))},{match.group(2)}", token))
            continue
        head = re.match(r"^(\d{1,4})DT$", clean)
        if head:
            head_value = _dedupe_overprint_digits(head.group(1))
            tails = [x for x in tokens if re.match(r"^[,.]\d{3}$", x.text) and -12 <= x.x0 - token.x1 <= 35 and abs(x.cy - token.cy) < 18]
            if tails:
                tail = min(tails, key=lambda x: abs(x.x0 - token.x1) + abs(x.cy - token.cy))
                amount = int(head_value)
                if amount <= 999:
                    found.append((f"{amount},{tail.text[1:]}", Token(clean, min(token.x0, tail.x0), min(token.top, tail.top), max(token.x1, tail.x1), max(token.bottom, tail.bottom))))
            elif token.bottom - token.top >= 20 and int(head_value) >= 20:
                found.append((f"{int(head_value)},000", token))
            continue
        if re.match(r"^\d{1,4}$", clean):
            currencies = [x for x in tokens if x.text.upper() == "DT" and -5 <= x.x0 - token.x1 <= 20 and abs(x.cy - token.cy) < 16]
            for currency in currencies[:1]:
                # A quantity digit has body-text height; a price head is
                # visibly larger than its DT marker. This prevents "4 Litres"
                # from borrowing the decimals of a nearby reference price.
                if token.bottom - token.top < (currency.bottom - currency.top) * 1.5:
                    continue
                tails = [x for x in tokens if re.match(r"^[,.]\d{3}$", x.text) and -16 <= x.x0 - currency.x1 <= 30 and abs(x.cy - currency.cy) < 18]
                if tails:
                    tail = min(tails, key=lambda x: abs(x.x0 - currency.x1) + abs(x.cy - currency.cy))
                    found.append((f"{int(clean)},{tail.text[1:]}", Token(clean, min(token.x0, tail.x0), min(token.top, tail.top), max(token.x1, tail.x1), max(token.bottom, tail.bottom))))
    unique: list[tuple[str, Token]] = []
    for value, token in found:
        if not any(abs(token.cx - other.cx) < 8 and abs(token.cy - other.cy) < 8 for _, other in unique):
            unique.append((value, token))
    return unique


def _region_for_anchor(anchor: Token, anchors: list[Token], width: float, height: float) -> tuple[float, float, float, float]:
    same_row = [a for a in anchors if abs(a.cy - anchor.cy) < max(32, height * .045)]
    lefts = [a.cx for a in same_row if a.cx < anchor.cx]
    rights = [a.cx for a in same_row if a.cx > anchor.cx]
    x0 = (max(lefts) + anchor.cx) / 2 if lefts else max(0, anchor.cx - width * .16)
    x1 = (min(rights) + anchor.cx) / 2 if rights else min(width, anchor.cx + width * .16)
    same_column = [a for a in anchors if abs(a.cx - anchor.cx) < width * .12 and abs(a.cy - anchor.cy) > 28]
    above = [a.cy for a in same_column if a.cy < anchor.cy]
    below = [a.cy for a in same_column if a.cy > anchor.cy]
    y0 = (max(above) + anchor.cy) / 2 if above else max(0, anchor.cy - height * .12)
    y1 = (min(below) + anchor.cy) / 2 if below else min(height, anchor.cy + height * .14)
    return max(0, x0 - 8), max(0, y0 - 8), min(width, x1 + 8), min(height, y1 + 8)


def _cluster_supported_edges(values: list[float], tolerance: float = 6.0) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        target = next((cluster for cluster in clusters if abs(np.mean(cluster) - value) <= tolerance), None)
        if target is None:
            clusters.append([value])
        else:
            target.append(value)
    return [float(np.mean(cluster)) for cluster in clusters if len(cluster) >= 2]


def _grid_region_for_anchor(page, anchor: Token, anchors: list[Token]) -> tuple[float, float, float, float]:
    # InDesign exports the visually subtle rounded card backgrounds as vector
    # curves. Their repeated bounds reveal the actual invisible catalogue grid.
    large_curves = [
        curve for curve in page.curves
        if curve.get("fill")
        and float(curve.get("width", 0)) >= page.width * .14
        and float(curve.get("height", 0)) >= page.height * .12
        and float(curve.get("x0", 0)) >= -3
        and float(curve.get("x1", 0)) <= page.width + 3
        and float(curve.get("top", 0)) >= -3
        and float(curve.get("bottom", 0)) <= page.height + 3
    ]
    x_edges = _cluster_supported_edges([float(curve[key]) for curve in large_curves for key in ("x0", "x1")])
    y_edges = _cluster_supported_edges([float(curve[key]) for curve in large_curves for key in ("top", "bottom")])
    left = max((edge for edge in x_edges if edge <= anchor.cx), default=None)
    right = min((edge for edge in x_edges if edge > anchor.cx), default=None)
    top = max((edge for edge in y_edges if edge <= anchor.cy), default=None)
    bottom = min((edge for edge in y_edges if edge > anchor.cy), default=None)
    if None not in (left, right, top, bottom) and right - left >= 55 and bottom - top >= 75:
        return max(0, left + 1), max(0, top + 1), min(page.width, right - 1), min(page.height, bottom - 1)
    return _region_for_anchor(anchor, anchors, page.width, page.height)


def _single_product_curve_region(page, anchor: Token, anchors: list[Token]) -> tuple[float, float, float, float] | None:
    candidates = []
    # Depending on the InDesign export, the same product frame is encoded
    # either as a Bézier curve or as a PDF rectangle. Rectangles are especially
    # common on the food pages and are the most reliable source of cell limits.
    vector_frames = list(page.curves) + list(page.rects)
    for frame in vector_frames:
        x0, x1 = float(frame.get("x0", 0)), float(frame.get("x1", 0))
        top, bottom = float(frame.get("top", 0)), float(frame.get("bottom", 0))
        width, height = x1 - x0, bottom - top
        if width < 55 or height < 70 or width > page.width * .62 or height > page.height * .62:
            continue
        if not (x0 <= anchor.cx <= x1 and top <= anchor.cy <= bottom):
            continue
        contained_prices = sum(x0 <= other.cx <= x1 and top <= other.cy <= bottom for other in anchors)
        if contained_prices != 1:
            continue
        candidates.append((width * height, (max(0, x0), max(0, top), min(page.width, x1), min(page.height, bottom))))
    if not candidates:
        return None
    seed_area, seed = min(candidates, key=lambda item: item[0])
    # A coloured feature card may sit behind a regular column-shaped clipping
    # path. If both contain the same single price, the enclosing vector is the
    # true product cell (e.g. a two-column pizza feature), not the narrow mask.
    wrappers = [
        (area, box) for area, box in candidates
        if seed_area * 1.2 <= area <= seed_area * 3.2
        and box[0] <= seed[0] + 8 and box[1] <= seed[1] + 8
        and box[2] >= seed[2] - 8 and box[3] >= seed[3] - 8
    ]
    return max(wrappers, key=lambda item: item[0])[1] if wrappers else seed


def _shared_product_curve_region(page, anchor: Token, anchors: list[Token]) -> tuple[float, float, float, float] | None:
    """Split a multi-offer feature frame using the prices inside the frame."""
    frames = []
    for frame in list(page.curves) + list(page.rects):
        x0, x1 = float(frame.get("x0", 0)), float(frame.get("x1", 0))
        top, bottom = float(frame.get("top", 0)), float(frame.get("bottom", 0))
        width, height = x1 - x0, bottom - top
        if width < 110 or height < 100 or width > page.width * .8 or height > page.height * .7:
            continue
        if not (x0 <= anchor.cx <= x1 and top <= anchor.cy <= bottom):
            continue
        inside = [a for a in anchors if x0 <= a.cx <= x1 and top <= a.cy <= bottom]
        # More than four anchors means this is a page-level decorative path,
        # not a shared product feature.
        if len(inside) < 2 or len(inside) > 4:
            continue
        frames.append((width * height, (x0, top, x1, bottom), inside))
    if not frames:
        return None
    _, (x0, top, x1, bottom), inside = min(frames, key=lambda item: item[0])
    row = sorted([a for a in inside if abs(a.cy - anchor.cy) <= max(35, (bottom - top) * .28)], key=lambda a: a.cx)
    if len(row) >= 2:
        index = row.index(anchor)
        left = (row[index - 1].cx + anchor.cx) / 2 if index else x0
        right = (anchor.cx + row[index + 1].cx) / 2 if index + 1 < len(row) else x1
        if right - left >= 55:
            return max(0, left), max(0, top), min(page.width, right), min(page.height, bottom)
    column = sorted([a for a in inside if abs(a.cx - anchor.cx) <= max(35, (x1 - x0) * .28)], key=lambda a: a.cy)
    if len(column) >= 2:
        index = column.index(anchor)
        upper = (column[index - 1].cy + anchor.cy) / 2 if index else top
        lower = (anchor.cy + column[index + 1].cy) / 2 if index + 1 < len(column) else bottom
        if lower - upper >= 70:
            return max(0, x0), max(0, upper), min(page.width, x1), min(page.height, lower)
    return None


def _partition_region_by_anchors(
    bbox: tuple[float, float, float, float], anchor: Token, anchors: list[Token]
) -> tuple[float, float, float, float]:
    """Split a grid cell when it contains several stacked or side-by-side offers."""
    x0, top, x1, bottom = bbox
    inside = [a for a in anchors if x0 <= a.cx <= x1 and top <= a.cy <= bottom]
    if len(inside) < 2:
        return bbox
    x_span = (max(a.cx for a in inside) - min(a.cx for a in inside)) / max(1, x1 - x0)
    y_span = (max(a.cy for a in inside) - min(a.cy for a in inside)) / max(1, bottom - top)
    if y_span >= x_span:
        ordered = sorted(inside, key=lambda a: a.cy)
        index = ordered.index(anchor)
        new_top = (ordered[index - 1].cy + anchor.cy) / 2 if index else top
        new_bottom = (anchor.cy + ordered[index + 1].cy) / 2 if index + 1 < len(ordered) else bottom
        return x0, new_top, x1, new_bottom
    ordered = sorted(inside, key=lambda a: a.cx)
    index = ordered.index(anchor)
    new_left = (ordered[index - 1].cx + anchor.cx) / 2 if index else x0
    new_right = (anchor.cx + ordered[index + 1].cx) / 2 if index + 1 < len(ordered) else x1
    return new_left, top, new_right, bottom


def _product_region(page, anchor: Token, anchors: list[Token]) -> tuple[tuple[float, float, float, float], str]:
    individual = _single_product_curve_region(page, anchor, anchors)
    if individual:
        return individual, "cadre_vectoriel"
    shared = _shared_product_curve_region(page, anchor, anchors)
    if shared:
        return shared, "cadre_partage"
    grid = _grid_region_for_anchor(page, anchor, anchors)
    adaptive = _region_for_anchor(anchor, anchors, page.width, page.height)
    if grid != adaptive:
        partitioned = _partition_region_by_anchors(grid, anchor, anchors)
        mode = "grille_partagee" if partitioned != grid else "grille_vectorielle"
        return partitioned, mode
    return adaptive, "zone_adaptative"


def _valid_product_region(page, anchor: Token, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    if not (0 <= anchor.cx <= page.width and 0 <= anchor.cy <= page.height):
        return False
    if x1 - x0 < page.width * .095 or bottom - top < page.height * .075:
        return False
    return True


def _group_lines(tokens: list[Token]) -> list[tuple[str, float]]:
    lines: list[list[Token]] = []
    clean_tokens = []
    for token in tokens:
        clean = token.text.replace(" ", "").upper()
        if re.match(r"^(?:D+T+|\d{1,4}D+T+|[,\.]\d{3})$", clean):
            continue
        clean_tokens.append(token)
    for token in sorted(clean_tokens, key=lambda t: (t.cy, t.x0)):
        line = next((row for row in reversed(lines[-5:]) if abs(np.mean([x.cy for x in row]) - token.cy) < 3.5), None)
        if line is None:
            lines.append([token])
        else:
            line.append(token)
    grouped = []
    for line in lines:
        ordered = [x for x in sorted(line, key=lambda t: t.x0) if x.text.strip()]
        if not ordered:
            continue
        text = ordered[0].text.strip()
        previous = ordered[0]
        for token in ordered[1:]:
            gap = token.x0 - previous.x1
            # Fragmented glyph runs created by InDesign touch or nearly touch;
            # real word spaces on the supplied catalogues are ~1.7-3 points.
            separator = "" if gap <= .9 else " "
            text += separator + token.text.strip()
            previous = token
        # Some PDFs contain every glyph twice at exactly the same position.
        # pdfplumber then returns PPrrééppaarraattiioonn instead of Préparation.
        words = []
        for word in text.split():
            if len(word) >= 2 and len(word) % 2 == 0 and all(word[i] == word[i + 1] for i in range(0, len(word), 2)):
                word = word[::2]
            words.append(word)
        grouped.append((" ".join(words), float(np.mean([x.cy for x in ordered]))))
    return grouped


def _cashback_promotion(tokens: list[Token]) -> str:
    for token in tokens:
        clean = token.text.replace(" ", "").upper()
        head = re.match(r"^\+(\d{1,3})D+T+$", clean)
        if not head:
            continue
        tails = [
            other for other in tokens
            if re.match(r"^[,.]\d{3}$", other.text)
            and -15 <= other.x0 - token.x1 <= 45
            and abs(other.cy - token.cy) < 24
        ]
        if tails:
            tail = min(tails, key=lambda other: abs(other.x0 - token.x1) + abs(other.cy - token.cy))
            return f"{int(head.group(1))},{tail.text[1:]} DT versés"
    return ""


def _text_fields(tokens: list[Token], price: str, anchor: Token | None = None) -> tuple[str, str, str, str, str, int]:
    lines = _group_lines(tokens)
    texts = [text for text, _ in lines]
    arabic = " ".join(x for x in texts if ARABIC.search(x))
    latin = [x for x in texts if not ARABIC.search(x) and not PRICE_HEAD.match(x.replace(" ", "")) and x.upper() != "DT" and not NOISE.search(x)]
    pourcentage = next((m.group(1) + " %" for x in texts if (m := PERCENT.match(x.replace(" ", "")))), "")
    joined = " ".join(texts).replace("ERFFO", "OFFRE")
    promotion = _cashback_promotion(tokens)
    if not promotion:
        # Promotional badges often place "+ article", OFFRE and GRATUIT(E) on
        # separate visual lines. Rebuild the phrase in reading order while
        # discarding the vertical OFFRE label and stray badge numerals.
        free_badge = ""
        plus_index = next((i for i, line in enumerate(texts) if re.match(r"^\s*\+\s*[A-Za-zÀ-ÿ]", line)), None)
        if plus_index is not None:
            gratuit_index = next((i for i in range(plus_index, min(len(texts), plus_index + 5)) if re.search(r"GRATUIT", texts[i], re.I)), None)
            if gratuit_index is not None:
                badge = " ".join(texts[plus_index:gratuit_index + 1]).replace("ERFFO", "OFFRE")
                suffix = re.search(r"GRATUIT(?:ES?|S)?", badge, re.I)
                item = re.sub(r"GRATUIT(?:ES?|S)?|\bOFFRE\b|\b\d+\b|^\s*\+", " ", badge, flags=re.I)
                item = re.sub(r"\s+", " ", item).strip()
                if item and suffix:
                    free_badge = f"+ {item.capitalize()} {suffix.group(0).lower()}"
        free_item = re.search(r"\+\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]{1,45}?)\s+(GRATUIT(?:ES?|S)?)\b", joined, re.I)
        plus = re.search(r"(\d+)\s*\+\s*(\d+)", joined)
        dont = re.search(r"DONT\s+(\d+)\s*(LITRES?|PI[EÈ]CES?|UNIT[EÉ]S?)?.*?GRATUIT", joined, re.I)
        if free_badge:
            promotion = free_badge
        elif free_item:
            item = re.sub(r"\s+", " ", free_item.group(1)).strip()
            promotion = f"+ {item.capitalize()} {free_item.group(2).lower()}"
        elif plus and re.search(r"GRATUIT", joined, re.I):
            promotion = f"{int(plus.group(1))} achetés + {int(plus.group(2))} gratuit"
        elif dont:
            unit = dont.group(2)
            promotion = f"{int(dont.group(1))} {unit.lower()} gratuit" if unit else f"{int(dont.group(1))} produit gratuit dans le lot"
        else:
            promotion_lines = [
                x.replace("ERFFO", "OFFRE") for x in latin
                if re.search(r"gratuit|offre|lot de|vers[ée]s|les \d+\b(?!\s*(?:g|kg|ml|cl|l|litres?)\b)", x, re.I)
            ]
            if any(re.search(r"gratuit", x, re.I) for x in promotion_lines):
                promotion_lines = [x for x in promotion_lines if not re.search(r"vers[ée]s", x, re.I)]
            promotion = " ".join(dict.fromkeys(promotion_lines))
    marque_match = next((re.search(r"[“\"]([^”\"]+)[”\"]", x) for x in latin if re.search(r"[“\"]([^”\"]+)[”\"]", x)), None)
    marque = marque_match.group(1) if marque_match else ""
    quantite_match = next((QUANTITY.search(x) for x in reversed(latin) if QUANTITY.search(x)), None)
    quantite = quantite_match.group(0) if quantite_match else ""
    candidates = [x for x in latin if len(x) > 2 and re.search(r"[A-Za-zÀ-ÿ]", x) and not re.search(r"[“”\"]", x) and not re.search(r"gratuit|offre|erffo|^\s*\+|\d+\s*\+\s*\d+|vers[ée]s|^\s*%|^\d+%?$|garantie|existe en|variétés|plusieurs|achat à crédit|chaud|froid|écran|litres?\s*$", x, re.I)]
    produit = "Produit à vérifier"
    if marque_match:
        brand_index = next((i for i, x in enumerate(latin) if marque_match.group(0) in x), -1)
        before = [x for x in latin[:brand_index] if x in candidates and x != marque]
        if before:
            produit = before[-1]
    if produit == "Produit à vérifier":
        ranked = sorted(candidates, key=lambda x: (bool(re.search(r"[“\"]", x)), len(x) < 4, len(x)))
        if ranked:
            produit = ranked[0]
    produit = re.sub(r"\s*%\s*$", "", produit).strip()
    confidence = 35 + (20 if produit != "Produit à vérifier" else 0) + (15 if marque else 0) + (10 if quantite else 0) + (10 if pourcentage else 0)
    return produit, arabic, marque, quantite, promotion, min(confidence, 95)


def _embedded_product_bbox(images: list[dict], region: tuple[float, float, float, float], anchor: Token, tokens: list[Token] | None = None) -> tuple[float, float, float, float] | None:
    x0, top, x1, bottom = region
    region_area = max(1, (x1 - x0) * (bottom - top))
    candidates = []
    for item in images:
        ix0, ix1 = float(item.get("x0", 0)), float(item.get("x1", 0))
        itop, ibottom = float(item.get("top", 0)), float(item.get("bottom", 0))
        width, height = ix1 - ix0, ibottom - itop
        cx, cy = (ix0 + ix1) / 2, (itop + ibottom) / 2
        area = width * height
        if not (x0 <= cx <= x1 and top <= cy <= bottom):
            continue
        if width < 22 or height < 22 or area < 700 or area > region_area * .72:
            continue
        if width / height > 4.5 or height / width > 4.5:
            continue
        distance = ((cx - anchor.cx) ** 2 + (cy - anchor.cy) ** 2) ** .5
        touches_edge = ix0 <= x0 + 4 or ix1 >= x1 - 4 or itop <= top + 4 or ibottom >= bottom - 4
        # Packshots tend to be interior assets. Tiled backgrounds frequently
        # touch a cell edge, so penalise them without discarding valid large packs.
        score = area - distance * 20 - (area * .78 if touches_edge else 0)
        candidates.append((score, area, touches_edge, (ix0, itop, ix1, ibottom)))
    if not candidates:
        return None
    seed = max(candidates, key=lambda value: value[0])
    _, seed_area, seed_edge, box = seed
    ux0, utop, ux1, ubottom = box
    if not seed_edge:
        for score, area, touches_edge, other in candidates:
            if other == box or touches_edge or area < seed_area * .42:
                continue
            ox0, otop, ox1, obottom = other
            overlap_x = max(0, min(ux1, ox1) - max(ux0, ox0))
            overlap_y = max(0, min(ubottom, obottom) - max(utop, otop))
            if overlap_x > 0 and overlap_y > min(ubottom - utop, obottom - otop) * .25:
                ux0, utop, ux1, ubottom = min(ux0, ox0), min(utop, otop), max(ux1, ox1), max(ubottom, obottom)
    if tokens:
        overlay_tops = [
            token.top for token in tokens
            if utop < token.top < ubottom
            and token.x1 > ux0 and token.x0 < ux1
            and re.search(r"[A-Za-zÀ-ÿ\u0600-\u06ff]", token.text)
            and not re.match(r"^(?:DT|D+T+|d.?économie)$", token.text, re.I)
        ]
        if overlay_tops and min(overlay_tops) - utop > 25:
            ubottom = min(ubottom, min(overlay_tops) - 2)
    return ux0, utop, ux1, ubottom


def _crop_images(page_image: Path, bbox: tuple[float, float, float, float], pdf_size: tuple[float, float], crop_path: Path, product_path: Path, product_bbox: tuple[float, float, float, float] | None = None) -> bool:
    image = Image.open(page_image).convert("RGB")
    sx, sy = image.width / pdf_size[0], image.height / pdf_size[1]
    box = tuple(int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(bbox))
    box = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
    crop = image.crop(box)
    crop.save(crop_path, quality=92)

    subject_box = product_bbox or bbox
    subject_pixels = tuple(int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(subject_box))
    # Exact cell crops must stop at the catalogue boundary; padding would leak
    # a thin strip of the neighbouring product. Keep padding only for an
    # explicitly detected inner packshot.
    pad = 10 if product_bbox is not None else 0
    subject_pixels = (max(0, subject_pixels[0] - pad), max(0, subject_pixels[1] - pad), min(image.width, subject_pixels[2] + pad), min(image.height, subject_pixels[3] + pad))
    crop = image.crop(subject_pixels)
    # Preserve the catalogue background exactly as printed. This is a crop,
    # not a background-removal/segmentation operation.
    crop.save(product_path, format="PNG")
    return product_bbox is None


def extract_pdf(source: Path, folder: Path, progress=None) -> list[Product]:
    pages = render_pdf(source, folder / "pages")
    from .pipeline import extract_offers
    return extract_offers(source, folder, pages, progress)


def import_image(source: Path, folder: Path) -> list[Product]:
    image = Image.open(source).convert("RGB")
    page_path = folder / "pages/page-01.jpg"
    image.save(page_path, quality=94)
    uid = uuid.uuid4().hex[:10]
    crop_rel, product_rel = f"crops/{uid}.jpg", f"products/{uid}.png"
    image.save(folder / crop_rel, quality=92)
    image.convert("RGBA").save(folder / product_rel)
    text = ""
    try:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True, use_textline_orientation=True, lang="fr")
        results = engine.predict(str(source))
        text = " ".join(str(item) for result in results for item in result.get("rec_texts", []))
    except (ImportError, RuntimeError, ValueError, AttributeError):
        text = "OCR local non installé"
    return [Product(id=uid, photo=product_rel, source_crop=crop_rel, produit=text or "OCR requis", page=1, confiance=0, statut="À vérifier", bbox=[0, 0, image.width, image.height], crop_mode="image_complete")]


def extract(source: Path, folder: Path, progress=None) -> list[Product]:
    if source.suffix.lower() == ".pdf":
        return extract_pdf(source, folder, progress)
    return import_image(source, folder)

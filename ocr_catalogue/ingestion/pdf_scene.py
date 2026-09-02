from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image

from ..domain import BBox, DocumentScene, PageScene, SemanticRole, VisualObject


ARABIC = re.compile(r"[\u0600-\u06ff]")


def _language(text: str) -> str:
    has_ar = bool(ARABIC.search(text))
    has_latin = bool(re.search(r"[A-Za-zÀ-ÿ]", text))
    if has_ar and has_latin:
        return "mixed"
    return "ar" if has_ar else "fr" if has_latin else "unknown"


def _collapse_overprint_word(word: str) -> str:
    # Two identical characters can be a legitimate value (11, 22, ...).
    # Overprint artefacts expose at least two duplicated glyph pairs, e.g.
    # DDTT, 1111 or PPrr; only collapse those longer sequences here.
    if len(word) >= 4 and len(word) % 2 == 0 and all(word[i] == word[i + 1] for i in range(0, len(word), 2)):
        return word[::2]
    return word


def _dedupe_words(words: list[dict]) -> list[dict]:
    unique: list[dict] = []
    for word in sorted(words, key=lambda item: (round(float(item["top"]), 1), round(float(item["x0"]), 1), str(item["text"]))):
        text = _collapse_overprint_word(str(word.get("text", "")).strip())
        if not text:
            continue
        current = dict(word)
        current["text"] = text
        duplicate = next((other for other in unique if other["text"] == text and abs(float(other["x0"]) - float(current["x0"])) < .8 and abs(float(other["top"]) - float(current["top"])) < .8), None)
        if duplicate is None:
            unique.append(current)
    return unique


def _word_objects(page_number: int, words: list[dict]) -> list[VisualObject]:
    objects = []
    for index, word in enumerate(words):
        bbox = BBox(float(word["x0"]), float(word["top"]), float(word["x1"]), float(word["bottom"]))
        objects.append(VisualObject(
            id=f"p{page_number}-w{index}", page=page_number, raw_type="word", bbox=bbox,
            text=str(word["text"]), font_size=float(word.get("size") or bbox.height),
            font_name=str(word.get("fontname") or ""), color=word.get("non_stroking_color"),
            language=_language(str(word["text"])), source_ids=[f"p{page_number}-w{index}"],
        ))
    return objects


def _line_objects(page_number: int, words: list[VisualObject]) -> list[VisualObject]:
    if not words:
        return []
    median_height = statistics.median(max(1.0, word.bbox.height) for word in words)
    rows: list[list[VisualObject]] = []
    for word in sorted(words, key=lambda item: (item.bbox.cy, item.bbox.x0)):
        tolerance = max(1.5, min(median_height * .42, word.bbox.height * .45))
        row = next((candidate for candidate in reversed(rows[-8:]) if abs(statistics.mean(item.bbox.cy for item in candidate) - word.bbox.cy) <= tolerance), None)
        if row is None:
            rows.append([word])
        else:
            row.append(word)
    lines = []
    line_index = 0
    for row in rows:
        ordered = sorted(row, key=lambda item: item.bbox.x0)
        typical = max(1.0, statistics.median([item.font_size for item in ordered]))
        # A physical line can contain several independent catalogue columns.
        # Word spacing scales with the local font; gaps larger than that scale
        # start a new semantic line instead of joining the whole page row.
        split_gap = typical * 2.15
        segments: list[list[VisualObject]] = [[ordered[0]]]
        for previous, current in zip(ordered, ordered[1:]):
            gap = current.bbox.x0 - previous.bbox.x1
            size_ratio = max(previous.font_size, current.font_size) / max(1.0, min(previous.font_size, current.font_size))
            # Price numerals can sit on the same baseline and very close to a
            # designation. Their abrupt type-scale change is a stronger
            # boundary signal than whitespace alone.
            if gap > split_gap or (gap >= -typical * .08 and size_ratio >= 1.8):
                segments.append([current])
            else:
                segments[-1].append(current)
        for segment in segments:
            chunks = [segment[0].text]
            for previous, current in zip(segment, segment[1:]):
                gap = current.bbox.x0 - previous.bbox.x1
                chunks.append(("" if gap <= typical * .08 else " ") + current.text)
            text = "".join(chunks).strip()
            bbox = segment[0].bbox
            for word in segment[1:]:
                bbox = bbox.union(word.bbox)
            lines.append(VisualObject(
                id=f"p{page_number}-l{line_index}", page=page_number, raw_type="line", bbox=bbox,
                text=text, font_size=max(item.font_size for item in segment),
                font_name=max((item.font_name for item in segment), key=lambda name: sum(1 for item in segment if item.font_name == name), default=""),
                language=_language(text), source_ids=[item.id for item in segment],
                metadata={"mean_font_size": statistics.mean(item.font_size for item in segment)},
            ))
            line_index += 1
    return lines


def _image_objects(page_number: int, images: list[dict], page_area: float) -> list[VisualObject]:
    grouped: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for image in images:
        key = tuple(round(float(image.get(name, 0))) for name in ("x0", "top", "x1", "bottom"))
        grouped[key].append(image)
    result = []
    for index, group in enumerate(grouped.values()):
        item = group[0]
        bbox = BBox(float(item.get("x0", 0)), float(item.get("top", 0)), float(item.get("x1", 0)), float(item.get("bottom", 0)))
        if bbox.area <= 4:
            continue
        result.append(VisualObject(
            id=f"p{page_number}-i{index}", page=page_number, raw_type="image", bbox=bbox,
            semantic_role=SemanticRole.IMAGE, semantic_confidence=.8, image_id=str(item.get("name") or item.get("stream") or index),
            metadata={"duplicates": len(group), "page_fraction": bbox.area / max(1, page_area)},
        ))
    return result


def _container_objects(page_number: int, page, word_height: float) -> list[VisualObject]:
    result = []
    frames = list(page.rects) + list(page.curves)
    for index, frame in enumerate(frames):
        bbox = BBox(float(frame.get("x0", 0)), float(frame.get("top", 0)), float(frame.get("x1", 0)), float(frame.get("bottom", 0))).clip(page.width, page.height)
        if bbox.width < word_height * 3 or bbox.height < word_height * 2 or bbox.area > page.width * page.height * .92:
            continue
        result.append(VisualObject(
            id=f"p{page_number}-c{index}", page=page_number, raw_type="container", bbox=bbox,
            color=frame.get("non_stroking_color") or frame.get("fill"), semantic_role=SemanticRole.CONTAINER,
            semantic_confidence=.45, metadata={"stroke": frame.get("stroke"), "fill": frame.get("fill")},
        ))
    return result


def _whitespace_runs(activity: np.ndarray, brightness: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    if activity.size == 0:
        return []
    activity_threshold = float(np.percentile(activity, 28))
    # Use the catalogue's own background distribution. This rejects flat
    # coloured artwork: uniformity alone is not whitespace.
    brightness_threshold = float(np.percentile(brightness, 68))
    mask = (activity <= activity_threshold) & (brightness >= brightness_threshold)
    runs = []
    start = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= min_run:
                runs.append((start, index))
            start = None
    return runs


def _visual_separators(page_number: int, raster_path: Path, width: float, height: float) -> list[VisualObject]:
    image = Image.open(raster_path).convert("L")
    image.thumbnail((800, 1200))
    array = np.asarray(image, dtype=np.float32)
    dx = np.abs(np.diff(array, axis=1, prepend=array[:, :1])).mean(axis=0)
    dy = np.abs(np.diff(array, axis=0, prepend=array[:1, :])).mean(axis=1)
    vertical = _whitespace_runs(dx, array.mean(axis=0), max(2, round(array.shape[1] * .004)))
    horizontal = _whitespace_runs(dy, array.mean(axis=1), max(2, round(array.shape[0] * .004)))
    sx, sy = width / array.shape[1], height / array.shape[0]
    result = []
    for index, (start, end) in enumerate(vertical):
        result.append(VisualObject(
            id=f"p{page_number}-sv{index}", page=page_number, raw_type="separator", semantic_role=SemanticRole.SEPARATOR,
            semantic_confidence=.35, bbox=BBox(start * sx, 0, end * sx, height), metadata={"orientation": "vertical"},
        ))
    for index, (start, end) in enumerate(horizontal):
        result.append(VisualObject(
            id=f"p{page_number}-sh{index}", page=page_number, raw_type="separator", semantic_role=SemanticRole.SEPARATOR,
            semantic_confidence=.35, bbox=BBox(0, start * sy, width, end * sy), metadata={"orientation": "horizontal"},
        ))
    return result


def extract_document_scene(source: Path, raster_pages: list[Path]) -> DocumentScene:
    pages = []
    with pdfplumber.open(source) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                extracted = page.extract_words(use_text_flow=False, keep_blank_chars=False, extra_attrs=["fontname", "size", "non_stroking_color"]) or []
            except Exception:
                try:
                    extracted = page.extract_words(use_text_flow=False, keep_blank_chars=False, extra_attrs=["fontname", "size"]) or []
                except Exception:
                    extracted = page.extract_words(use_text_flow=False, keep_blank_chars=False) or []
            # InDesign PDFs may retain objects from the neighbouring spread
            # outside the CropBox. They are not visible on the rendered page
            # and must never participate in offer association.
            extracted = [
                item for item in extracted
                if 0 <= (float(item.get("x0", 0)) + float(item.get("x1", 0))) / 2 <= page.width
                and 0 <= (float(item.get("top", 0)) + float(item.get("bottom", 0))) / 2 <= page.height
            ]
            word_objects = _word_objects(page_index + 1, _dedupe_words(extracted))
            lines = _line_objects(page_index + 1, word_objects)
            median_height = statistics.median([obj.bbox.height for obj in word_objects]) if word_objects else 8.0
            images = [
                obj for obj in _image_objects(page_index + 1, page.images, page.width * page.height)
                if 0 <= obj.bbox.cx <= page.width and 0 <= obj.bbox.cy <= page.height
            ]
            containers = [
                obj for obj in _container_objects(page_index + 1, page, median_height)
                if 0 <= obj.bbox.cx <= page.width and 0 <= obj.bbox.cy <= page.height
            ]
            raster = raster_pages[page_index]
            separators = _visual_separators(page_index + 1, raster, page.width, page.height)
            pages.append(PageScene(
                number=page_index + 1, width=page.width, height=page.height,
                objects=word_objects + lines + images + containers,
                separators=separators, raster_path=str(raster),
            ))
    return DocumentScene(source=str(source), pages=pages)

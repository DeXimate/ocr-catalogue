from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_catalogue.domain import DocumentScene, PageScene, SemanticRole
from ocr_catalogue.graph import build_spatial_graph
from ocr_catalogue.ingestion.pdf_scene import _container_objects, _dedupe_words, _image_objects, _line_objects, _word_objects
from ocr_catalogue.offers.resolver import _assign_objects, _offer_candidates
from ocr_catalogue.semantics import classify_document
from ocr_catalogue.style import infer_catalogue_style


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("page", type=int)
    args = parser.parse_args()
    with pdfplumber.open(args.source) as pdf:
        source_page = pdf.pages[args.page - 1]
        raw = source_page.extract_words(use_text_flow=False, keep_blank_chars=False, extra_attrs=["fontname", "size"])
        words = _word_objects(args.page, _dedupe_words(raw))
        lines = _line_objects(args.page, words)
        images = _image_objects(args.page, source_page.images, source_page.width * source_page.height)
        containers = _container_objects(args.page, source_page, 8)
        page = PageScene(args.page, source_page.width, source_page.height, words + lines + images + containers)
    document = DocumentScene(str(args.source), [page])
    classify_document(document)
    infer_catalogue_style(document)
    graph = build_spatial_graph(page, document.style)
    candidates = _offer_candidates(page, graph)
    _assign_objects(page, graph, candidates)
    objects = page.object_by_id()
    facts = {fact.id: fact for fact in page.numeric_facts}
    for candidate in candidates:
        main = [facts[item].value for item in candidate.numeric_ids if item in facts and facts[item].role.value == "PRICE_MAIN"]
        assigned = [
            (objects[item].semantic_role.value, objects[item].text, objects[item].bbox.as_list(), objects[item].metadata.get("page_fraction"))
            for item in candidate.object_ids if item in objects
        ]
        print("OFFER", main, assigned)


if __name__ == "__main__":
    main()

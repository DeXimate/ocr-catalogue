from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .domain import Offer
from .ingestion import extract_document_scene
from .models import Product
from .offers import resolve_document_offers
from .semantics import classify_document
from .style import infer_catalogue_style


def _crop_offer(offer: Offer, page_image: Path, page_size: tuple[float, float], folder: Path) -> tuple[str, str]:
    image = Image.open(page_image).convert("RGB")
    sx, sy = image.width / page_size[0], image.height / page_size[1]
    bbox = offer.bbox.clip(*page_size)
    pixels = (
        max(0, round(bbox.x0 * sx)), max(0, round(bbox.top * sy)),
        min(image.width, round(bbox.x1 * sx)), min(image.height, round(bbox.bottom * sy)),
    )
    if pixels[2] - pixels[0] < 8 or pixels[3] - pixels[1] < 8:
        raise ValueError(f"Limites d'offre invalides page {offer.page}: {offer.bbox}")
    crop = image.crop(pixels)
    crop_rel = f"crops/{offer.id}.jpg"
    product_rel = f"products/{offer.id}.png"
    crop.save(folder / crop_rel, quality=94)
    crop.save(folder / product_rel, format="PNG")
    return crop_rel, product_rel


def _to_product(offer: Offer, crop_rel: str, product_rel: str) -> Product:
    confidence = max(0, min(99, round(offer.confidence * 100)))
    return Product(
        id=offer.id, photo=product_rel, source_crop=crop_rel,
        produit=offer.product_name, designation_ar=offer.arabic_name,
        marque=offer.brand, modele=offer.model, quantite=offer.quantity,
        prix_promo=offer.main_price, ancien_prix=offer.reference_price,
        remise=offer.discount, promotion=offer.promotion,
        cashback=offer.cashback, price_basis=offer.price_basis,
        specifications=offer.technical_specs, raisons_revision=offer.review_reasons,
        page=offer.page, confiance=confidence,
        statut="À vérifier" if offer.review_reasons or confidence < 88 else "Validé",
        bbox=offer.bbox.as_list(), crop_mode=offer.crop_mode,
    )


def extract_offers(source: Path, folder: Path, raster_pages: list[Path], progress=None) -> list[Product]:
    document = extract_document_scene(source, raster_pages)
    classify_document(document)
    infer_catalogue_style(document)
    offers = resolve_document_offers(document)
    products = []
    page_sizes = {page.number: (page.width, page.height) for page in document.pages}
    for index, offer in enumerate(offers):
        crop_rel, product_rel = _crop_offer(offer, raster_pages[offer.page - 1], page_sizes[offer.page], folder)
        products.append(_to_product(offer, crop_rel, product_rel))
        if progress and (index == len(offers) - 1 or offers[index + 1].page != offer.page):
            progress(offer.page, len(document.pages))
    audit = {
        "engine": "offer-graph-v1",
        "style": asdict(document.style),
        "offers": [asdict(offer) for offer in offers],
    }
    (folder / "analysis.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    return products

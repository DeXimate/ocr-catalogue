from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

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
        prix_promo=offer.main_price, pourcentage=offer.percentage,
        promotion=offer.promotion,
        cashback=offer.cashback, price_basis=offer.price_basis,
        specifications=offer.technical_specs, raisons_revision=offer.review_reasons,
        page=offer.page, confiance=confidence,
        statut="À vérifier" if offer.review_reasons or confidence < 88 else "Validé",
        bbox=offer.bbox.as_list(), crop_mode=offer.crop_mode,
        region_quality=offer.region_quality,
    )


def _diagnostic_tile(product: Product, folder: Path, size: tuple[int, int] = (320, 260)) -> Image.Image:
    source = folder / product.source_crop
    image = Image.open(source).convert("RGB")
    canvas = Image.new("RGB", size, "white")
    visual_height = size[1] - 38
    image.thumbnail((size[0] - 12, visual_height - 10))
    x = (size[0] - image.width) // 2
    y = (visual_height - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, visual_height, size[0], size[1]), fill=(246, 247, 248))
    label = f"p.{product.page}  {product.produit[:34]}  {product.prix_promo}"
    draw.text((8, visual_height + 11), label, fill=(30, 34, 38))
    return ImageOps.expand(canvas, border=1, fill=(215, 218, 221))


def _write_diagnostic_contact_sheets(products: list[Product], folder: Path) -> list[str]:
    diagnostic_dir = folder / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    pages: dict[int, list[Product]] = {}
    for product in products:
        pages.setdefault(product.page, []).append(product)
    written = []
    representative = []
    for page, page_products in sorted(pages.items()):
        ordered = sorted(page_products, key=lambda product: (product.confiance, product.id))
        if len(ordered) <= 6:
            sample = ordered
        else:
            indexes = sorted({round(index * (len(ordered) - 1) / 5) for index in range(6)})
            sample = [ordered[index] for index in indexes]
        tiles = [_diagnostic_tile(product, folder) for product in sample]
        sheet = Image.new("RGB", (322 * 3, 262 * 2), (238, 240, 242))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % 3) * 322, (index // 3) * 262))
        relative = f"diagnostics/page-{page:02d}-samples.jpg"
        sheet.save(folder / relative, quality=90)
        written.append(relative)
        representative.append(ordered[0])
    if representative:
        tiles = [_diagnostic_tile(product, folder, (260, 215)) for product in representative]
        columns = min(4, len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        overview = Image.new("RGB", (262 * columns, 217 * rows), (238, 240, 242))
        for index, tile in enumerate(tiles):
            overview.paste(tile, ((index % columns) * 262, (index // columns) * 217))
        overview.save(diagnostic_dir / "all-pages-sample.jpg", quality=90)
        written.append("diagnostics/all-pages-sample.jpg")
    return written


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
    diagnostic_images = _write_diagnostic_contact_sheets(products, folder)
    audit = {
        "engine": "offer-region-v3-native-panels",
        "style": asdict(document.style),
        "offers": [asdict(offer) for offer in offers],
        "pages": [
            {
                "page": page.number,
                "offer_count": sum(offer.page == page.number for offer in offers),
                "accepted_count": sum(offer.page == page.number and offer.region_quality.get("accepted", False) for offer in offers),
                "offers": [
                    {
                        "id": offer.id,
                        "page": offer.page,
                        "product": offer.product_name,
                        "region": offer.bbox.as_list(),
                        "safe_region": offer.safe_bbox,
                        "crop_mode": offer.crop_mode,
                        "quality": offer.region_quality,
                    }
                    for offer in offers if offer.page == page.number
                ],
            }
            for page in document.pages
        ],
        "diagnostic_images": diagnostic_images,
    }
    (folder / "analysis.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    (folder / "crop_diagnostics.json").write_text(json.dumps({"engine": audit["engine"], "pages": audit["pages"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return products

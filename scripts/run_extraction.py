from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_catalogue.engines import extract


def main() -> None:
    parser = argparse.ArgumentParser(description="Exécute le moteur OFFER dans un dossier isolé.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for name in ("pages", "crops", "products"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    products = extract(args.source, args.output)
    (args.output / "products.json").write_text(
        json.dumps([product.to_dict() for product in products], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TOTAL {len(products)}")
    for product in products:
        if product.marque in {"JUDY", "ALWAYS", "PAMPERS"} or product.produit in {
            "Couches bébé", "Préparation pâte à pizza", "Yaourt aromatisé", "Saucisson à l’ail", "Poulet rôti"
        }:
            print(product.produit, product.marque, product.prix_promo, product.pourcentage, product.page, product.bbox, product.crop_mode, product.photo)


if __name__ == "__main__":
    main()

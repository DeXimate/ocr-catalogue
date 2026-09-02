from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_catalogue import storage
from ocr_catalogue.engines import extract


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalcule un catalogue existant avec le moteur OCR courant.")
    parser.add_argument("job_id")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    folder = storage.job_folder(args.job_id)
    job = storage.load_job(args.job_id)
    job.pop("error", None)
    job.update(status="Traitement", progress=0)
    storage.save_job(args.job_id, job)

    products = extract(args.source, folder)
    job = storage.load_job(args.job_id)
    job.update(status="Terminé", progress=100, products=[product.to_dict() for product in products], error="")
    storage.save_job(args.job_id, job)

    print(f"TOTAL {len(products)}")
    for product in products:
        if product.produit in {"Couches bébé", "Préparation pâte à pizza", "Yaourt aromatisé", "Saucisson à l’ail", "Poulet rôti"}:
            print(product.produit, product.prix_promo, product.promotion, product.page, product.bbox, product.photo)


if __name__ == "__main__":
    main()

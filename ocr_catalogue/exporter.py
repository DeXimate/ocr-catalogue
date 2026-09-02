from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Font, PatternFill


COLUMNS = [
    ("photo", "Photo"), ("produit", "Produit"), ("marque", "Marque"),
    ("quantite", "Quantité"), ("prix_promo", "Prix promo"),
    ("pourcentage", "Pourcentage"),
    ("promotion", "Promotion"), ("page", "Page"),
    ("confiance", "Confiance"), ("statut", "Statut"),
]


def _selected(products: list[dict], scope: str) -> list[dict]:
    if scope == "validated":
        return [p for p in products if p.get("statut") == "Validé"]
    if scope == "selected":
        return [p for p in products if p.get("selected")]
    return products


def export_csv(products: list[dict], folder: Path, target: Path, include_photos: bool, scope: str) -> None:
    columns = COLUMNS if include_photos else [c for c in COLUMNS if c[0] != "photo"]
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[label for _, label in columns], extrasaction="ignore")
        writer.writeheader()
        for product in _selected(products, scope):
            row = {label: product.get(key, "") for key, label in columns}
            if include_photos:
                row["Photo"] = product.get("photo", "")
            writer.writerow(row)


def export_xlsx(products: list[dict], folder: Path, target: Path, include_photos: bool, scope: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Produits"
    columns = COLUMNS if include_photos else [c for c in COLUMNS if c[0] != "photo"]
    ws.append([label for _, label in columns])
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="C8192E")
    rows = _selected(products, scope)
    for row_index, product in enumerate(rows, start=2):
        ws.append(["" if key == "photo" else product.get(key, "") for key, _ in columns])
        if include_photos and product.get("photo"):
            path = folder / product["photo"]
            if path.exists():
                image = ExcelImage(str(path))
                image.thumbnail(76, 66)
                ws.add_image(image, f"A{row_index}")
                ws.row_dimensions[row_index].height = 54
    widths = {"A": 14, "B": 34, "C": 20, "D": 18, "E": 16, "F": 16, "G": 12, "H": 28}
    for key, width in widths.items():
        ws.column_dimensions[key].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    review = wb.create_sheet("À vérifier")
    review.append([label for key, label in columns if key != "photo"])
    for product in rows:
        if product.get("statut") != "Validé":
            review.append([product.get(key, "") for key, _ in columns if key != "photo"])
    meta = wb.create_sheet("Métadonnées")
    meta.append(["Champ", "Valeur"])
    meta.append(["Nombre de produits", len(rows)])
    meta.append(["Photos incluses", "Oui" if include_photos else "Non"])
    meta.append(["Périmètre", scope])
    wb.save(target)

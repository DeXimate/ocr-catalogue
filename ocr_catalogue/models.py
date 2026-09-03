from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Product:
    id: str
    photo: str = ""
    source_crop: str = ""
    produit: str = ""
    designation_ar: str = ""
    marque: str = ""
    modele: str = ""
    quantite: str = ""
    caracteristiques: str = ""
    variantes_prix: str = ""
    prix_promo: str = ""
    pourcentage: str = ""
    promotion: str = ""
    cashback: str = ""
    price_basis: str = ""
    specifications: list[str] = field(default_factory=list)
    raisons_revision: list[str] = field(default_factory=list)
    page: int = 1
    confiance: int = 0
    statut: str = "À vérifier"
    selected: bool = False
    bbox: list[float] = field(default_factory=list)
    crop_mode: str = ""
    region_quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Product":
        value = dict(value)
        if not value.get("pourcentage") and value.get("remise"):
            value["pourcentage"] = value["remise"]
        value.pop("ancien_prix", None)
        value.pop("remise", None)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in allowed if key in value})

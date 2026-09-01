# OCR Catalogue Monoprix

Application locale Windows pour extraire les produits et prix promotionnels de catalogues PDF/JPG/PNG, corriger les résultats dans un tableau et exporter en XLSX ou CSV.

## Démarrage

```powershell
.\run.ps1
```

Puis ouvrir <http://127.0.0.1:8765>.

`launch.ps1` démarre le serveur en arrière-plan et ouvre automatiquement le navigateur par défaut. Il est utilisé par le raccourci du Bureau.

Le script utilise en priorité le runtime Python fourni avec Codex. Un Python 3.11+ avec `pypdf`, `pdfplumber`, `Pillow`, `numpy` et `openpyxl` convient également.

## Fonctionnalités

- extraction des textes et coordonnées des PDF InDesign ;
- détection des prix tunisiens à trois décimales ;
- association spatiale produit-prix et score de confiance ;
- aperçu de l'encart source et photo produit recadrée ;
- édition, sélection et validation en masse ;
- export XLSX avec images embarquées ou CSV avec chemins relatifs ;
- stockage local dans `data/` et reprise des traitements.

Les PDF scannés et images sans texte natif sont importés et signalés pour revue. Le connecteur OCR local est prévu dans `ocr_catalogue/engines.py`; PaddleOCR/Tesseract peuvent y être activés sans modifier l'interface.

Pour activer PaddleOCR et l'analyse locale des pages scannées :

```powershell
.\setup-advanced.ps1
```

Les photos sont toujours les cellules produit complètes, recadrées rectangulairement avec leur fond, leur prix et leur offre. Aucun arrière-plan n'est supprimé. Sans l'installation avancée, l'application utilise le texte natif et les structures vectorielles des PDF.

## Tests

```powershell
.\test.ps1
```

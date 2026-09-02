import csv
import unittest
from pathlib import Path

from openpyxl import load_workbook

from ocr_catalogue.engines import Token, _cashback_promotion, _dedupe_overprint_digits, _embedded_product_bbox, _group_lines, _merge_price_tokens, _partition_region_by_anchors, _text_fields, _valid_product_region
from ocr_catalogue.exporter import export_csv, export_xlsx
from ocr_catalogue.models import Product


class ExtractionTests(unittest.TestCase):
    def test_quantity_digit_cannot_borrow_price_decimals(self):
        tokens = [
            Token("4", 0, 0, 6, 13), Token("Litres", 8, 0, 40, 13),
            Token("6", 45, 10, 56, 38), Token("DT", 59, 11, 67, 21), Token(",090", 56, 18, 82, 37),
        ]
        self.assertEqual(_merge_price_tokens(tokens), [("6,090", Token("6", 45, 10, 82, 38))])

    def test_stacked_offers_split_a_shared_grid_cell(self):
        upper = Token("5DT", 200, 70, 230, 90)
        lower = Token("27DT", 175, 170, 205, 190)
        bbox = (145, 40, 280, 230)
        self.assertEqual(_partition_region_by_anchors(bbox, upper, [upper, lower]), (145, 40, 280, 130.0))
        self.assertEqual(_partition_region_by_anchors(bbox, lower, [upper, lower]), (145, 130.0, 280, 230))

    def test_truncated_spread_cell_is_rejected(self):
        class Page:
            width = 592
            height = 848
        self.assertFalse(_valid_product_region(Page(), Token("9DT", 595, 50, 615, 75), (552, 0, 592, 200)))

    def test_tunisian_price_is_preserved(self):
        tokens = [Token("16DT", 1, 1, 28, 20), Token(",990", 20, 10, 45, 24)]
        self.assertEqual(_merge_price_tokens(tokens)[0][0], "16,990")

    def test_numbers_without_currency_are_not_prices(self):
        tokens = [Token("400", 1, 1, 20, 12), Token("500", 21, 1, 40, 12)]
        self.assertEqual(_merge_price_tokens(tokens), [])

    def test_overprinted_digits_are_collapsed(self):
        self.assertEqual(_dedupe_overprint_digits("1188"), "18")
        self.assertEqual(_dedupe_overprint_digits("1100"), "10")
        self.assertEqual(_dedupe_overprint_digits("129"), "129")

    def test_cashback_amount_is_preserved(self):
        tokens = [
            Token("+0DT", 10, 10, 38, 30),
            Token(",500", 32, 19, 59, 35),
            Token("VERSÉS", 12, 40, 55, 50),
        ]
        self.assertEqual(_cashback_promotion(tokens), "0,500 DT versés")

    def test_adjacent_promotion_types_are_not_merged(self):
        values = [Token(x, 0, i * 10, 70, i * 10 + 8) for i, x in enumerate(["Nettoyant sols", "Versés", "OFFRE", "DONT 1 GRATUIT"])]
        self.assertEqual(_text_fields(values, "7,990")[4], "1 produit gratuit dans le lot")

    def test_free_unit_is_understood(self):
        values = [Token(x, 0, i * 10, 100, i * 10 + 8) for i, x in enumerate(["Eau de javel", "OFFRE", "DONT 1 LITRE GRATUIT", "“JUDY”"])]
        self.assertEqual(_text_fields(values, "4,190")[4], "1 litre gratuit")

    def test_free_item_name_is_preserved(self):
        values = [Token(x, 0, i * 10, 140, i * 10 + 8) for i, x in enumerate(["OFFRE", "+ LINGETTES BÉBÉ", "GRATUITES", "Couches bébé", "“LILAS”"])]
        produit, _, _, _, promotion, _ = _text_fields(values, "33,990")
        self.assertEqual(produit, "Couches bébé")
        self.assertEqual(promotion, "+ Lingettes bébé gratuites")

    def test_multiline_free_badge_is_reconstructed(self):
        values = [
            Token("+ LINGETTES 3", 0, 0, 100, 8),
            Token("ERFFO BÉBÉ", 0, 10, 100, 18),
            Token("GRATUITES", 0, 20, 100, 28),
            Token("Couches bébé", 0, 40, 100, 48),
            Token("“LILAS”", 0, 50, 100, 58),
        ]
        produit, _, _, _, promotion, _ = _text_fields(values, "33,990")
        self.assertEqual(produit, "Couches bébé")
        self.assertEqual(promotion, "+ Lingettes bébé gratuites")

    def test_price_basis_is_not_a_promotion(self):
        values = [
            Token("LES 100 G", 0, 0, 80, 8),
            Token("Saucisson à l’ail", 0, 20, 120, 28),
            Token("“EL MAZRAA”", 0, 30, 100, 38),
        ]
        _, _, _, _, promotion, _ = _text_fields(values, "1,990")
        self.assertEqual(promotion, "")

    def test_overprinted_words_are_collapsed(self):
        values = [
            Token("PPrrééppaarraattiioonn", 0, 0, 120, 8),
            Token("ppââttee", 125, 0, 175, 8),
            Token("àà", 180, 0, 190, 8),
            Token("ppiizzzzaa", 195, 0, 250, 8),
        ]
        self.assertEqual(_group_lines(values)[0][0], "Préparation pâte à pizza")

    def test_offer_quantity_is_understood(self):
        values = [Token(x, 0, i * 10, 90, i * 10 + 8) for i, x in enumerate(["OFFRE 2+1", "GRATUIT", "Tampons éponge", "“VILEDA”"])]
        produit, _, marque, _, promotion, _ = _text_fields(values, "4,990")
        self.assertEqual(produit, "Tampons éponge")
        self.assertEqual(marque, "VILEDA")
        self.assertEqual(promotion, "2 achetés + 1 gratuit")

    def test_kerning_fragments_are_rejoined(self):
        tokens = [
            Token("B", 0, 0, 5, 10), Token("lo", 5.5, 0, 14, 10), Token("cs", 14.5, 0, 23, 10),
            Token("cu", 29, 0, 38, 10), Token("ve", 38.5, 0, 47, 10), Token("t", 47.5, 0, 51, 10), Token("te", 51.5, 0, 59, 10),
        ]
        from ocr_catalogue.engines import _group_lines
        self.assertEqual(_group_lines(tokens)[0][0], "Blocs cuvette")

    def test_fields(self):
        values = [Token(x, 0, i * 10, 50, i * 10 + 8) for i, x in enumerate(["Lessive liquide", "“DIXAN”", "3 Litres", "54%"])]
        produit, _, marque, quantite, _, confidence = _text_fields(values, "16,990")
        self.assertEqual(produit, "Lessive liquide")
        self.assertEqual(marque, "DIXAN")
        self.assertIn("3 Litres", quantite)
        self.assertGreaterEqual(confidence, 80)

    def test_embedded_product_image_is_selected(self):
        images = [
            {"x0": 0, "x1": 200, "top": 0, "bottom": 200},
            {"x0": 20, "x1": 90, "top": 50, "bottom": 140},
            {"x0": 150, "x1": 190, "top": 60, "bottom": 100},
        ]
        result = _embedded_product_bbox(images, (0, 0, 120, 180), Token("13DT", 30, 10, 60, 30))
        self.assertEqual(result, (20.0, 50.0, 90.0, 140.0))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.products = [{"produit": "Lessive", "prix_promo": "16,990 DT", "photo": "products/x.png", "statut": "Validé", "selected": True}]
        self.folder = Path("data/test-output")
        self.folder.mkdir(parents=True, exist_ok=True)

    def test_csv_without_photo_column(self):
        target = self.folder / "out.csv"
        export_csv(self.products, self.folder, target, False, "all")
        with target.open(encoding="utf-8-sig") as f:
            self.assertNotIn("Photo", next(csv.reader(f)))

    def test_old_job_maps_remise_and_discards_old_price(self):
        product = Product.from_dict({"id": "old", "remise": "32 %", "ancien_prix": "14,990 DT"})
        self.assertEqual(product.pourcentage, "32 %")
        self.assertNotIn("ancien_prix", product.to_dict())
        self.assertNotIn("remise", product.to_dict())

    def test_xlsx_has_price_promo(self):
        target = self.folder / "out.xlsx"
        export_xlsx(self.products, self.folder, target, False, "all")
        wb = load_workbook(target)
        headers = [cell.value for cell in wb["Produits"][1]]
        self.assertIn("Prix promo", headers)
        self.assertIn("Pourcentage", headers)
        self.assertNotIn("Ancien prix", headers)
        self.assertNotIn("Remise", headers)
        self.assertNotIn("Photo", headers)


if __name__ == "__main__":
    unittest.main()

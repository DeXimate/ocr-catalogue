import unittest

from ocr_catalogue.domain import BBox, DocumentScene, NumericFact, NumericRole, PageScene, SemanticRole, VisualObject
from ocr_catalogue.graph import build_spatial_graph
from ocr_catalogue.ingestion.pdf_scene import _collapse_overprint_word, _line_objects
from ocr_catalogue.offers.resolver import _partition_container
from ocr_catalogue.semantics.classifier import _find_prices, _classify_lines
from ocr_catalogue.style import infer_catalogue_style


def word(identifier, text, x0, top, x1, bottom, size=10):
    return VisualObject(identifier, 1, "word", BBox(x0, top, x1, bottom), text=text, font_size=size, font_name="Test")


class OfferEngineTests(unittest.TestCase):
    def test_large_gap_creates_independent_semantic_lines(self):
        words = [word("a", "Produit A", 0, 0, 40, 10), word("b", "Produit B", 160, 0, 200, 10)]
        self.assertEqual([line.text for line in _line_objects(1, words)], ["Produit A", "Produit B"])

    def test_large_price_type_is_split_from_nearby_product_name(self):
        product = word("product", "Natte de plage", 0, 10, 75, 22, 12)
        price = word("price", "18", 78, 0, 104, 38, 38)
        self.assertEqual([line.text for line in _line_objects(1, [product, price])], ["Natte de plage", "18"])

    def test_quantity_cannot_supply_price_milliemes(self):
        words = [
            word("q", "4", 0, 20, 6, 33, 13), word("unit", "Litres", 8, 20, 40, 33, 13),
            word("head", "6", 45, 10, 56, 38, 28), word("dt", "DT", 59, 11, 67, 21, 10),
            word("comma", ",", 56, 18, 59, 37, 19), word("tail", "090", 59, 18, 82, 37, 19),
        ]
        lines = _line_objects(1, words)
        page = PageScene(1, 300, 500, words + lines)
        _classify_lines(page)
        prices = _find_prices(page)
        self.assertEqual([(price.value, price.role) for price in prices], [("6,090", NumericRole.PRICE_MAIN)])

    def test_discount_badge_cannot_borrow_neighbour_currency(self):
        words = [
            word("discount", "32", 100, 100, 124, 140, 36),
            word("percent", "%", 125, 118, 136, 136, 18),
            word("foreign-dt", "DT", 190, 110, 201, 123, 12),
        ]
        page = PageScene(1, 300, 500, words + _line_objects(1, words))
        self.assertEqual(_find_prices(page), [])

    def test_legitimate_two_digit_value_is_not_collapsed(self):
        self.assertEqual(_collapse_overprint_word("11"), "11")
        self.assertEqual(_collapse_overprint_word("1111"), "11")

    def test_regular_variant_is_not_a_product_name(self):
        product = word("product", "Tongs de plage", 0, 0, 80, 12, 12)
        product.font_name = "Arial-Bold"
        variant = word("variant", "Femme", 0, 14, 35, 24, 10)
        variant.font_name = "Arial-Regular"
        page = PageScene(1, 300, 500, [product, variant] + _line_objects(1, [product, variant]))
        _classify_lines(page)
        roles = {line.text: line.semantic_role for line in page.objects if line.raw_type == "line"}
        self.assertEqual(roles["Tongs de plage"], SemanticRole.PRODUCT_TEXT)
        self.assertEqual(roles["Femme"], SemanticRole.TECHNICAL_SPEC)

    def test_diagonal_offer_does_not_force_rectangular_cut(self):
        container = BBox(0, 0, 300, 300)
        result = _partition_container(container, (150, 150), [(50, 50)])
        self.assertEqual(result, container)

    def test_catalogue_style_is_recomputed_from_document(self):
        pages = []
        for number in (1, 2, 3):
            line = VisualObject(f"l{number}", number, "line", BBox(0, 480, 300, 500), text="Photos non contractuelles", font_size=6, semantic_role=SemanticRole.PRODUCT_TEXT)
            price_word = VisualObject(f"w{number}", number, "word", BBox(20, 20, 50, 55), text="9DT", font_size=35, font_name="Price")
            page = PageScene(number, 300, 500, [line, price_word])
            page.numeric_facts = [NumericFact(f"p{number}", number, "9DT", "9,000", price_word.bbox, NumericRole.PRICE_MAIN, .8, [price_word.id])]
            pages.append(page)
        document = DocumentScene("test.pdf", pages)
        style = infer_catalogue_style(document)
        self.assertIn("PHOTOS NON CONTRACTUELLES", style.repeated_noise)
        self.assertEqual(style.price_fonts, ["Price"])


if __name__ == "__main__":
    unittest.main()

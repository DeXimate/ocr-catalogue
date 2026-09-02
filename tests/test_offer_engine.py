import unittest

from ocr_catalogue.domain import BBox, DocumentScene, NumericFact, NumericRole, Offer, PageScene, SemanticRole, VisualObject
from ocr_catalogue.graph import build_spatial_graph
from ocr_catalogue.ingestion.pdf_scene import _collapse_overprint_word, _line_objects
from ocr_catalogue.offers.resolver import _offer_bbox, _partition_container
from ocr_catalogue.pipeline import _to_product
from ocr_catalogue.semantics.classifier import _classify_lines, _classify_non_price_numbers, _find_prices, parse_promotion
from ocr_catalogue.style import infer_catalogue_style


def word(identifier, text, x0, top, x1, bottom, size=10):
    return VisualObject(identifier, 1, "word", BBox(x0, top, x1, bottom), text=text, font_size=size, font_name="Test")


class OfferEngineTests(unittest.TestCase):
    def promotion(self, text):
        return VisualObject(
            "promo", 1, "line", BBox(0, 0, 200, 20), text=text, font_size=10,
            semantic_role=SemanticRole.PROMOTION, semantic_confidence=.9,
        )

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

    def test_percentage_maps_to_product_without_old_price(self):
        offer = Offer("offer", 1, BBox(0, 0, 100, 100), product_name="Tongs", main_price="9,990 DT", percentage="32 %")
        product = _to_product(offer, "crop.jpg", "photo.png")
        self.assertEqual(product.pourcentage, "32 %")
        self.assertEqual(product.prix_promo, "9,990 DT")
        self.assertNotIn("ancien_prix", product.to_dict())

    def test_red_badge_percentage_is_extracted_directly(self):
        number = word("discount", "32", 100, 100, 124, 140, 36)
        marker = word("percent", "%", 125, 118, 136, 136, 18)
        page = PageScene(1, 300, 500, [number, marker] + _line_objects(1, [number, marker]))
        percentages = [fact for fact in _classify_non_price_numbers(page) if fact.role == NumericRole.DISCOUNT]
        self.assertEqual([fact.value for fact in percentages], ["32"])

    def test_dimensions_are_not_a_promotion(self):
        self.assertEqual(parse_promotion([self.promotion("23 x 15 x 7 + 28 x 20 x 8 cm")]), "")

    def test_capacity_composition_is_not_a_promotion(self):
        self.assertEqual(parse_promotion([self.promotion("0,5 + 1 + 2 litres 3 Pièces")]), "")

    def test_product_composition_is_not_a_promotion(self):
        self.assertEqual(parse_promotion([self.promotion("+ 2 faitouts avec couvercle")]), "")

    def test_classifier_marks_compositions_as_technical_not_promotional(self):
        lines = [
            VisualObject("d", 1, "line", BBox(0, 0, 200, 12), text="23 x 15 x 7 + 28 x 20 x 8 cm", font_size=10, font_name="Bold"),
            VisualObject("c", 1, "line", BBox(0, 20, 200, 32), text="0,5 + 1 + 2 litres 3 Pièces", font_size=10, font_name="Bold"),
            VisualObject("p", 1, "line", BBox(0, 40, 200, 52), text="+ 2 faitouts avec couvercle", font_size=10, font_name="Bold"),
        ]
        page = PageScene(1, 300, 500, lines)
        _classify_lines(page)
        self.assertTrue(all(line.semantic_role == SemanticRole.TECHNICAL_SPEC for line in lines))

    def test_classifier_accepts_only_explicit_free_mechanism(self):
        line = VisualObject("promo", 1, "line", BBox(0, 0, 100, 12), text="2+1 GRATUIT", font_size=10, font_name="Bold")
        page = PageScene(1, 300, 500, [line])
        _classify_lines(page)
        self.assertEqual(line.semantic_role, SemanticRole.PROMOTION)

    def test_free_ratio_is_a_promotion(self):
        self.assertEqual(parse_promotion([self.promotion("2+1 GRATUIT")]), "2 achetés + 1 gratuit")

    def test_offered_article_is_a_promotion(self):
        self.assertEqual(parse_promotion([self.promotion("+ 1 article offert")]), "+ 1 article offert")

    def test_cashback_is_a_promotion(self):
        self.assertEqual(parse_promotion([], "0,500 DT versés"), "0,500 DT versés")

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

    def test_shared_raster_is_partitioned_and_footer_is_excluded(self):
        shared = VisualObject(
            "shared", 1, "image", BBox(0, 0, 300, 180),
            semantic_role=SemanticRole.IMAGE, metadata={"page_fraction": .36},
        )
        footer = VisualObject(
            "footer", 1, "line", BBox(0, 150, 300, 170),
            text="Les articles sont disponibles dans les magasins suivants",
            font_size=5, semantic_role=SemanticRole.HEADER_FOOTER,
        )
        product = VisualObject(
            "product", 1, "line", BBox(128, 42, 172, 55), text="Tabouret", font_size=11,
            semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9,
        )
        brand = VisualObject(
            "brand", 1, "line", BBox(132, 58, 168, 70), text="“SYFAX”", font_size=9,
            semantic_role=SemanticRole.BRAND, semantic_confidence=.9,
        )
        middle_price = NumericFact("p-middle", 1, "10,900 DT", "10,900", BBox(138, 82, 166, 110), NumericRole.PRICE_MAIN, .95)
        left_price = NumericFact("p-left", 1, "16,900 DT", "16,900", BBox(38, 82, 66, 110), NumericRole.PRICE_MAIN, .95)
        right_price = NumericFact("p-right", 1, "29,900 DT", "29,900", BBox(238, 82, 266, 110), NumericRole.PRICE_MAIN, .95)
        page = PageScene(1, 300, 500, [shared, footer, product, brand], numeric_facts=[left_price, middle_price, right_price])
        candidate = type("Candidate", (), {"id": "middle", "bbox": middle_price.bbox})()
        cores = {
            "left": BBox(30, 40, 75, 112), "middle": BBox(125, 40, 175, 112), "right": BBox(225, 40, 275, 112),
        }
        region, _ = _offer_bbox(page, candidate, [product, brand], [middle_price], cores)
        self.assertLess(region.x0, 150)
        self.assertGreater(region.x1, 150)
        self.assertGreater(region.x0, left_price.bbox.cx)
        self.assertLess(region.x1, right_price.bbox.cx)
        self.assertLessEqual(region.bottom, footer.bbox.top)
        self.assertFalse(region.contains_point(left_price.bbox.cx, left_price.bbox.cy))
        self.assertFalse(region.contains_point(right_price.bbox.cx, right_price.bbox.cy))

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

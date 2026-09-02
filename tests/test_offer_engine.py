import unittest

from ocr_catalogue.domain import BBox, DocumentScene, NumericFact, NumericRole, Offer, OfferCandidate, PageScene, SemanticRole, VisualObject
from ocr_catalogue.graph import build_spatial_graph
from ocr_catalogue.ingestion.pdf_scene import _collapse_overprint_word, _line_objects
from ocr_catalogue.offers.panel_detector import detect_native_panels
from ocr_catalogue.offers.resolver import _merge_product_with_priced_brand, _offer_bbox, _offer_candidates, _partition_container, _pick_product, _reassign_secondary_facts
from ocr_catalogue.offers.region_solver import build_offer_nuclei, solve_page_regions
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

    def test_credit_and_appliance_specs_do_not_seed_products(self):
        lines = [
            VisualObject("duration", 1, "line", BBox(0, 0, 90, 12), text="18 mois 36 mois", font_size=10, font_name="Bold"),
            VisualObject("programs", 1, "line", BBox(0, 20, 130, 32), text="13 COUVERTS 5 PROGRAMMES", font_size=10, font_name="Bold"),
            VisualObject("temperature", 1, "line", BBox(0, 40, 100, 52), text="CHAUD/FROID", font_size=10, font_name="Bold"),
            VisualObject("credit", 1, "line", BBox(0, 60, 120, 72), text="ACHAT À CRÉDIT", font_size=10, font_name="Bold"),
            VisualObject("frost", 1, "line", BBox(0, 80, 120, 92), text="NO FROST INVERTER", font_size=10, font_name="Bold"),
        ]
        page = PageScene(1, 300, 500, lines)
        _classify_lines(page)
        self.assertTrue(all(line.semantic_role == SemanticRole.TECHNICAL_SPEC for line in lines))

    def test_classifier_accepts_only_explicit_free_mechanism(self):
        line = VisualObject("promo", 1, "line", BBox(0, 0, 100, 12), text="2+1 GRATUIT", font_size=10, font_name="Bold")
        page = PageScene(1, 300, 500, [line])
        _classify_lines(page)
        self.assertEqual(line.semantic_role, SemanticRole.PROMOTION)

    def test_small_bottom_legal_band_is_header_footer(self):
        body = VisualObject("body", 1, "line", BBox(10, 50, 120, 62), text="Produit principal", font_size=10, font_name="Bold")
        footer = VisualObject("footer", 1, "line", BBox(5, 95, 195, 99), text="Liste des magasins participants", font_size=4, font_name="Regular")
        page = PageScene(1, 200, 100, [body, footer])

        _classify_lines(page)

        self.assertEqual(footer.semantic_role, SemanticRole.HEADER_FOOTER)

    def test_free_ratio_is_a_promotion(self):
        self.assertEqual(parse_promotion([self.promotion("2+1 GRATUIT")]), "2 achetés + 1 gratuit")

    def test_offered_article_is_a_promotion(self):
        self.assertEqual(parse_promotion([self.promotion("+ 1 article offert")]), "+ 1 article offert")

    def test_damaged_free_fragments_are_not_promotions(self):
        self.assertEqual(parse_promotion([self.promotion("FF GRATUIT")]), "")
        self.assertEqual(parse_promotion([self.promotion("+1 O gratuit")]), "")

    def test_classifier_rejects_damaged_free_fragments(self):
        lines = [
            VisualObject("ff", 1, "line", BBox(0, 0, 90, 12), text="FF GRATUIT", font_size=10, font_name="Bold"),
            VisualObject("one-letter", 1, "line", BBox(0, 40, 100, 52), text="+1 O gratuit", font_size=10, font_name="Bold"),
        ]
        page = PageScene(1, 300, 500, lines)
        _classify_lines(page)
        self.assertTrue(all(line.semantic_role != SemanticRole.PROMOTION for line in lines))

    def test_cashback_is_a_promotion(self):
        self.assertEqual(parse_promotion([], "0,500 DT versés"), "0,500 DT versés")

    def test_plus_prefixed_cashback_amount_is_extracted(self):
        words = [
            word("head", "+7", 20, 10, 42, 40, 24),
            word("currency", "DT", 44, 17, 53, 29, 10),
            word("tail", ",500", 53, 25, 78, 39, 13),
            word("mechanism", "VERSÉS", 35, 42, 76, 54, 10),
        ]
        page = PageScene(1, 200, 150, words + _line_objects(1, words))

        cashback = [fact for fact in _find_prices(page) if fact.role == NumericRole.CASHBACK]

        self.assertEqual([(fact.value, fact.role) for fact in cashback], [("7,500", NumericRole.CASHBACK)])

    def test_plus_prefixed_whole_dinar_cashback_is_extracted(self):
        words = [
            word("head", "+2", 20, 10, 42, 40, 24),
            word("currency", "DT", 44, 17, 53, 29, 10),
            word("mechanism", "VERSÉS", 24, 42, 70, 54, 10),
        ]
        page = PageScene(1, 200, 150, words + _line_objects(1, words))

        cashback = [fact for fact in _find_prices(page) if fact.role == NumericRole.CASHBACK]

        self.assertEqual([fact.value for fact in cashback], ["2,000"])

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

    def test_leading_connective_cannot_replace_bold_product_name(self):
        product = VisualObject(
            "product", 1, "line", BBox(20, 20, 90, 34), text="Shampooing", font_size=12,
            font_name="Bold", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.8,
        )
        continuation = VisualObject(
            "continuation", 1, "line", BBox(20, 36, 140, 50), text="ou après shampooing", font_size=12,
            font_name="Bold", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9,
        )
        style = type("Style", (), {"body_font_size": 10, "product_fonts": ["Bold"]})()
        price = NumericFact("price", 1, "16,990 DT", "16,990", BBox(160, 20, 195, 55), NumericRole.PRICE_MAIN, .95)

        self.assertEqual(_pick_product([product, continuation], price, style), "Shampooing")

    def test_diagonal_offer_does_not_force_rectangular_cut(self):
        container = BBox(0, 0, 300, 300)
        result = _partition_container(container, (150, 150), [(50, 50)])
        self.assertEqual(result, container)

    def test_native_panel_detector_uses_exclusive_pdf_card(self):
        left_product = VisualObject(
            "left-product", 1, "line", BBox(15, 25, 80, 40),
            text="Tabouret", semantic_role=SemanticRole.PRODUCT_TEXT,
            semantic_confidence=.95,
        )
        right_product = VisualObject(
            "right-product", 1, "line", BBox(115, 25, 180, 40),
            text="Parasol", semantic_role=SemanticRole.PRODUCT_TEXT,
            semantic_confidence=.95,
        )
        left_price = NumericFact(
            "left-price", 1, "16,900 DT", "16,900",
            BBox(55, 70, 88, 96), NumericRole.PRICE_MAIN, .96,
        )
        right_price = NumericFact(
            "right-price", 1, "89,900 DT", "89,900",
            BBox(155, 70, 188, 96), NumericRole.PRICE_MAIN, .96,
        )
        left_card = VisualObject(
            "left-card", 1, "container", BBox(5, 5, 98, 110),
            semantic_role=SemanticRole.CONTAINER,
        )
        right_card = VisualObject(
            "right-card", 1, "container", BBox(102, 5, 195, 110),
            semantic_role=SemanticRole.CONTAINER,
        )
        page = PageScene(
            1, 200, 130,
            [left_product, right_product, left_card, right_card],
            [left_price, right_price],
        )
        candidates = [
            OfferCandidate("left", 1, [left_product.id], [left_price.id], left_price.bbox, .95),
            OfferCandidate("right", 1, [right_product.id], [right_price.id], right_price.bbox, .95),
        ]
        nuclei = build_offer_nuclei(page, candidates)
        panels = detect_native_panels(
            page,
            candidates,
            nuclei,
            {"left": left_price.bbox, "right": right_price.bbox},
        )

        self.assertEqual(panels["left"].bbox, left_card.bbox)
        self.assertEqual(panels["right"].bbox, right_card.bbox)

        solutions = solve_page_regions(page, candidates)
        self.assertEqual(solutions["left"].mode, "panel_native")
        self.assertEqual(solutions["left"].region, left_card.bbox)
        self.assertEqual(solutions["right"].region, right_card.bbox)

    def test_shared_native_panel_is_not_used_as_one_offer(self):
        left_product = VisualObject(
            "left-product", 1, "line", BBox(15, 25, 80, 40),
            text="Produit A", semantic_role=SemanticRole.PRODUCT_TEXT,
            semantic_confidence=.95,
        )
        right_product = VisualObject(
            "right-product", 1, "line", BBox(115, 25, 180, 40),
            text="Produit B", semantic_role=SemanticRole.PRODUCT_TEXT,
            semantic_confidence=.95,
        )
        left_price = NumericFact("left-price", 1, "9,990 DT", "9,990", BBox(50, 70, 85, 95), NumericRole.PRICE_MAIN, .95)
        right_price = NumericFact("right-price", 1, "8,990 DT", "8,990", BBox(150, 70, 185, 95), NumericRole.PRICE_MAIN, .95)
        shared = VisualObject("shared", 1, "container", BBox(5, 5, 195, 110), semantic_role=SemanticRole.CONTAINER)
        page = PageScene(1, 200, 130, [left_product, right_product, shared], [left_price, right_price])
        candidates = [
            OfferCandidate("left", 1, [left_product.id], [left_price.id], left_price.bbox, .95),
            OfferCandidate("right", 1, [right_product.id], [right_price.id], right_price.bbox, .95),
        ]
        nuclei = build_offer_nuclei(page, candidates)
        panels = detect_native_panels(page, candidates, nuclei, {"left": left_price.bbox, "right": right_price.bbox})
        self.assertEqual(panels, {})

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

    def test_page_region_solver_is_complete_exclusive_and_respects_footer(self):
        objects = []
        facts = []
        candidates = []
        for index, centre in enumerate((50, 150, 250)):
            product = VisualObject(
                f"product-{index}", 1, "line", BBox(centre - 28, 35, centre + 28, 49),
                text=f"Produit {index}", font_size=11,
                semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.95,
            )
            brand = VisualObject(
                f"brand-{index}", 1, "line", BBox(centre - 20, 52, centre + 20, 63),
                text=f"MARQUE {index}", font_size=9,
                semantic_role=SemanticRole.BRAND, semantic_confidence=.9,
            )
            price = NumericFact(
                f"price-{index}", 1, "9,990 DT", "9,990", BBox(centre - 18, 75, centre + 18, 104),
                NumericRole.PRICE_MAIN, .96,
            )
            objects += [product, brand]
            facts.append(price)
            candidates.append(OfferCandidate(
                f"offer-{index}", 1, [product.id, brand.id], [price.id], price.bbox, .95,
            ))
        objects += [
            VisualObject("shared", 1, "image", BBox(0, 0, 300, 175), semantic_role=SemanticRole.IMAGE, metadata={"page_fraction": .35}),
            VisualObject("footer", 1, "line", BBox(0, 180, 300, 198), text="Mentions légales", semantic_role=SemanticRole.HEADER_FOOTER),
        ]
        separators = [
            VisualObject("sep-1", 1, "separator", BBox(99, 0, 101, 180), semantic_role=SemanticRole.SEPARATOR, metadata={"orientation": "vertical"}),
            VisualObject("sep-2", 1, "separator", BBox(199, 0, 201, 180), semantic_role=SemanticRole.SEPARATOR, metadata={"orientation": "vertical"}),
        ]
        page = PageScene(1, 300, 240, objects, facts, separators)
        nuclei = build_offer_nuclei(page, candidates)
        solutions = solve_page_regions(page, candidates)
        for candidate in candidates:
            solution = solutions[candidate.id]
            self.assertTrue(solution.region.contains(nuclei[candidate.id]))
            self.assertLessEqual(solution.region.bottom, 180)
            self.assertEqual(solution.quality["semantic_coverage"], 1.0)
            self.assertEqual(solution.quality["competing_price_centres"], 0)
            self.assertFalse(solution.quality["crosses_header_footer"])
        for index, candidate in enumerate(candidates):
            for other_index, price in enumerate(facts):
                if index != other_index:
                    self.assertFalse(solutions[candidate.id].region.contains_point(price.bbox.cx, price.bbox.cy))
        self.assertFalse(solutions["offer-0"].region.intersects(solutions["offer-1"].region))
        self.assertFalse(solutions["offer-1"].region.intersects(solutions["offer-2"].region))

    def test_card_boundary_wins_over_internal_image_caption_whitespace(self):
        upper_product = VisualObject("upper-product", 1, "line", BBox(30, 70, 110, 88), text="Produit haut", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        lower_product = VisualObject("lower-product", 1, "line", BBox(30, 205, 110, 223), text="Produit bas", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        upper_price = NumericFact("upper-price", 1, "9,990 DT", "9,990", BBox(115, 65, 150, 92), NumericRole.PRICE_MAIN, .95)
        lower_price = NumericFact("lower-price", 1, "8,990 DT", "8,990", BBox(115, 200, 150, 227), NumericRole.PRICE_MAIN, .95)
        separators = [
            VisualObject("card-edge", 1, "separator", BBox(0, 118, 200, 122), semantic_role=SemanticRole.SEPARATOR, metadata={"orientation": "horizontal"}),
            VisualObject("internal-gap", 1, "separator", BBox(0, 178, 200, 182), semantic_role=SemanticRole.SEPARATOR, metadata={"orientation": "horizontal"}),
        ]
        page = PageScene(1, 200, 300, [upper_product, lower_product], [upper_price, lower_price], separators)
        candidates = [
            OfferCandidate("upper", 1, [upper_product.id], [upper_price.id], upper_price.bbox, .9),
            OfferCandidate("lower", 1, [lower_product.id], [lower_price.id], lower_price.bbox, .9),
        ]

        solutions = solve_page_regions(page, candidates)

        self.assertAlmostEqual(solutions["lower"].safe_region.top, 120)

    def test_unpriced_product_nuclei_survive_when_another_price_exists(self):
        priced = VisualObject("priced", 1, "line", BBox(10, 10, 70, 24), text="Produit vendu", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        unpriced = VisualObject("unpriced", 1, "line", BBox(210, 210, 280, 224), text="Produit sans prix", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        price = NumericFact("price", 1, "9,990 DT", "9,990", BBox(18, 35, 58, 65), NumericRole.PRICE_MAIN, .95)
        page = PageScene(1, 300, 300, [priced, unpriced], [price])
        graph = build_spatial_graph(page, type("Style", (), {"body_font_size": 10})())
        candidates = _offer_candidates(page, graph)
        self.assertTrue(any(candidate.numeric_ids == [price.id] for candidate in candidates))
        self.assertTrue(any(candidate.object_ids == [unpriced.id] and "prix_principal_absent" in candidate.contradictions for candidate in candidates))

    def test_percentage_badge_is_reassigned_inside_its_offer_territory(self):
        left_product = VisualObject(
            "left-product", 1, "line", BBox(20, 80, 75, 94), text="Produit gauche",
            semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.95,
        )
        right_product = VisualObject(
            "right-product", 1, "line", BBox(125, 80, 180, 94), text="Produit droite",
            semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.95,
        )
        left_price = NumericFact("left-price", 1, "9,990 DT", "9,990", BBox(28, 105, 62, 132), NumericRole.PRICE_MAIN, .95)
        right_price = NumericFact("right-price", 1, "8,990 DT", "8,990", BBox(138, 105, 172, 132), NumericRole.PRICE_MAIN, .95)
        left_discount = NumericFact("left-discount", 1, "32 %", "32", BBox(70, 102, 92, 127), NumericRole.DISCOUNT, .9)
        page = PageScene(1, 200, 220, [left_product, right_product], [left_price, right_price, left_discount])
        candidates = [
            OfferCandidate("left", 1, [left_product.id], [left_price.id], left_price.bbox, .9),
            # Simulate a bad first graph assignment: the left badge belongs to the right offer.
            OfferCandidate("right", 1, [right_product.id], [right_price.id, left_discount.id], right_price.bbox, .9),
        ]

        _reassign_secondary_facts(page, candidates)

        self.assertIn(left_discount.id, candidates[0].numeric_ids)
        self.assertNotIn(left_discount.id, candidates[1].numeric_ids)

    def test_cashback_label_moves_with_amount_to_card_territory(self):
        upper = VisualObject("upper", 1, "line", BBox(30, 55, 100, 70), text="Produit haut", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        lower = VisualObject("lower", 1, "line", BBox(30, 145, 100, 160), text="Produit bas", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        label = VisualObject("label", 1, "line", BBox(55, 112, 95, 124), text="VERSÉS", font_size=10, semantic_role=SemanticRole.PROMOTION, semantic_confidence=.9)
        upper_price = NumericFact("upper-price", 1, "9,990 DT", "9,990", BBox(110, 50, 145, 75), NumericRole.PRICE_MAIN, .95)
        lower_price = NumericFact("lower-price", 1, "8,990 DT", "8,990", BBox(110, 140, 145, 165), NumericRole.PRICE_MAIN, .95)
        cashback = NumericFact("cashback", 1, "+2 DT", "2,000", BBox(55, 100, 90, 116), NumericRole.CASHBACK, .95)
        separator = VisualObject("edge", 1, "separator", BBox(0, 88, 200, 92), semantic_role=SemanticRole.SEPARATOR, metadata={"orientation": "horizontal"})
        page = PageScene(1, 200, 220, [upper, lower, label], [upper_price, lower_price, cashback], [separator])
        candidates = [
            OfferCandidate("upper-offer", 1, [upper.id, label.id], [upper_price.id, cashback.id], upper_price.bbox, .9),
            OfferCandidate("lower-offer", 1, [lower.id], [lower_price.id], lower_price.bbox, .9),
        ]

        _reassign_secondary_facts(page, candidates)

        self.assertIn(cashback.id, candidates[1].numeric_ids)
        self.assertIn(label.id, candidates[1].object_ids)
        self.assertNotIn(label.id, candidates[0].object_ids)

    def test_cashback_does_not_replace_neighbouring_free_offer(self):
        television = VisualObject("tv", 1, "line", BBox(145, 35, 195, 50), text="Téléviseur", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        chips = VisualObject("chips", 1, "line", BBox(80, 160, 130, 175), text="Chips", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        ratio = VisualObject("ratio", 1, "line", BBox(55, 125, 80, 137), text="1+1", semantic_role=SemanticRole.PROMOTION, semantic_confidence=.9)
        free = VisualObject("free", 1, "line", BBox(55, 138, 90, 150), text="GRATUIT", semantic_role=SemanticRole.PROMOTION, semantic_confidence=.9)
        label = VisualObject("label", 1, "line", BBox(100, 95, 130, 107), text="VERSÉS", semantic_role=SemanticRole.PROMOTION, semantic_confidence=.9)
        tv_price = NumericFact("tv-price", 1, "399 DT", "399,000", BBox(170, 55, 205, 80), NumericRole.PRICE_MAIN, .95)
        chips_price = NumericFact("chips-price", 1, "4,790 DT", "4,790", BBox(85, 130, 125, 155), NumericRole.PRICE_MAIN, .95)
        cashback = NumericFact("cashback", 1, "+30 DT", "30,000", BBox(95, 78, 130, 103), NumericRole.CASHBACK, .95)
        page = PageScene(1, 240, 220, [television, chips, ratio, free, label], [tv_price, chips_price, cashback])
        candidates = [
            OfferCandidate("tv-offer", 1, [television.id], [tv_price.id], tv_price.bbox, .9),
            OfferCandidate("chips-offer", 1, [chips.id, ratio.id, free.id, label.id], [chips_price.id, cashback.id], chips_price.bbox, .9),
        ]

        _reassign_secondary_facts(page, candidates)

        self.assertIn(cashback.id, candidates[0].numeric_ids)
        self.assertNotIn(cashback.id, candidates[1].numeric_ids)
        self.assertIn(label.id, candidates[0].object_ids)

    def test_cashback_prefers_coherent_priced_offer_over_unpriced_fragment(self):
        priced_product = VisualObject("priced-product", 1, "line", BBox(145, 35, 205, 50), text="Téléviseur", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        unpriced_fragment = VisualObject("fragment", 1, "line", BBox(100, 58, 130, 72), text="QLED", semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        label = VisualObject("label", 1, "line", BBox(92, 105, 125, 117), text="VERSÉS", semantic_role=SemanticRole.PROMOTION, semantic_confidence=.9)
        price = NumericFact("price", 1, "399 DT", "399,000", BBox(170, 55, 210, 82), NumericRole.PRICE_MAIN, .95)
        cashback = NumericFact("cashback", 1, "+30 DT", "30,000", BBox(90, 82, 128, 107), NumericRole.CASHBACK, .95)
        page = PageScene(1, 240, 220, [priced_product, unpriced_fragment, label], [price, cashback])
        candidates = [
            OfferCandidate("priced", 1, [priced_product.id], [price.id], price.bbox, .9),
            OfferCandidate("fragment", 1, [unpriced_fragment.id, label.id], [cashback.id], unpriced_fragment.bbox, .4),
        ]

        _reassign_secondary_facts(page, candidates)

        self.assertIn(cashback.id, candidates[0].numeric_ids)
        self.assertNotIn(cashback.id, candidates[1].numeric_ids)

    def test_orphan_product_line_merges_with_immediately_aligned_priced_brand(self):
        product = VisualObject("product", 1, "line", BBox(100, 40, 140, 54), text="QLED", font_size=14, semantic_role=SemanticRole.PRODUCT_TEXT, semantic_confidence=.9)
        brand = VisualObject("brand", 1, "line", BBox(100, 57, 170, 71), text="“MAXWELL”", font_size=14, semantic_role=SemanticRole.BRAND, semantic_confidence=.9)
        price = NumericFact("price", 1, "399 DT", "399,000", BBox(180, 35, 220, 70), NumericRole.PRICE_MAIN, .95)
        page = PageScene(1, 240, 220, [product, brand], [price])
        candidates = [
            OfferCandidate("priced", 1, [brand.id], [price.id], price.bbox, .9),
            OfferCandidate("orphan", 1, [product.id], [], product.bbox, .4),
        ]

        _merge_product_with_priced_brand(page, candidates)

        self.assertEqual(len(candidates), 1)
        self.assertIn(product.id, candidates[0].object_ids)

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

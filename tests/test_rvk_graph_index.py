import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.core.data_models import LlmKeywordAnalysis
from src.utils.clients.rvk_graph_index import (
    GraphCandidate,
    GraphEvidence,
    RvkGraphIndex,
)


class RvkGraphIndexTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._temp_dir.name)
        self.index = RvkGraphIndex(data_dir=self.data_dir)

    def tearDown(self):
        self._temp_dir.cleanup()

    def _write_mock_graph_index(self):
        conn = self.index._connect()
        self.index._create_schema(conn)
        now_iso = datetime.now().replace(tzinfo=None).isoformat()
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", self.index.SCHEMA_VERSION),
                ("release", "test"),
                ("built_at", now_iso),
                ("observed_max_ancestor_depth", "17"),
                ("default_max_ancestor_hops", "17"),
                ("default_max_descendant_hops", "17"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO rvk_node
            (notation, label, branch_family, ancestor_path, depth_from_root, normalized_depth, semantic_text, is_leaf)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "RF 34000",
                    "Allgemeine Umweltgeografie",
                    "R",
                    "Geografie > Umweltgeografie",
                    2,
                    0.12,
                    "Allgemeine Umweltgeografie | Geografie > Umweltgeografie",
                    0,
                ),
                (
                    "RF 34600",
                    "Bodenkunde",
                    "R",
                    "Geografie > Umweltgeografie > Bodenkunde",
                    3,
                    0.18,
                    "Bodenkunde | Geografie > Umweltgeografie > Bodenkunde",
                    0,
                ),
                (
                    "RF 34650",
                    "Bodenchemie",
                    "R",
                    "Geografie > Umweltgeografie > Bodenkunde > Chemische Prozesse",
                    4,
                    0.24,
                    "Bodenchemie | Geografie > Umweltgeografie > Bodenkunde > Chemische Prozesse",
                    1,
                ),
                (
                    "RF 34699",
                    "Bodenkontamination",
                    "R",
                    "Geografie > Umweltgeografie > Bodenkunde > Schadstoffe",
                    4,
                    0.24,
                    "Bodenkontamination | Geografie > Umweltgeografie > Bodenkunde > Schadstoffe",
                    1,
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO rvk_edge (parent_notation, child_notation, edge_type) VALUES (?, ?, ?)",
            [
                ("RF 34000", "RF 34600", "parent_child"),
                ("RF 34600", "RF 34650", "parent_child"),
                ("RF 34600", "RF 34699", "parent_child"),
            ],
        )
        conn.executemany(
            "INSERT INTO rvk_closure (ancestor_notation, descendant_notation, distance) VALUES (?, ?, ?)",
            [
                ("RF 34000", "RF 34000", 0),
                ("RF 34600", "RF 34600", 0),
                ("RF 34650", "RF 34650", 0),
                ("RF 34699", "RF 34699", 0),
                ("RF 34000", "RF 34600", 1),
                ("RF 34600", "RF 34650", 1),
                ("RF 34600", "RF 34699", 1),
                ("RF 34000", "RF 34650", 2),
                ("RF 34000", "RF 34699", 2),
            ],
        )
        conn.executemany(
            "INSERT INTO concept_node (gnd_id, preferred_label, normalized_label) VALUES (?, ?, ?)",
            [
                ("4029921-1", "Cadmium", "cadmium"),
                ("4007394-4", "Boeden", "boeden"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO concept_rvk_edge
            (gnd_id, notation, field_tag, register_term, direct_weight)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("4029921-1", "RF 34699", "750", "Cadmium", 1.0),
                ("4007394-4", "RF 34600", "750", "Boeden", 1.0),
            ],
        )
        conn.executemany(
            "INSERT INTO term_node (term_norm, display_term, term_kind) VALUES (?, ?, ?)",
            [
                ("cadmium", "Cadmium", "register"),
                ("boeden", "Boeden", "register"),
                ("bodenkontamination", "Bodenkontamination", "label"),
                ("bodenchemie", "Bodenchemie", "label"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO term_rvk_edge
            (term_norm, notation, source_field, weight)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("cadmium", "RF 34699", "register", 1.0),
                ("boeden", "RF 34600", "register", 1.0),
                ("boeden", "RF 34650", "register", 0.8),
                ("bodenkontamination", "RF 34699", "label", 0.85),
                ("bodenchemie", "RF 34650", "label", 0.85),
            ],
        )
        conn.commit()
        conn.close()

    def test_ensure_graph_creates_schema_and_meta(self):
        self.assertTrue(self.index.ensure_graph())
        self.assertTrue(self.index.db_path.exists())

        conn = sqlite3.connect(self.index.db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view', 'virtual table')"
            ).fetchall()
        }
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        conn.close()

        self.assertIn("rvk_node", tables)
        self.assertIn("rvk_edge", tables)
        self.assertIn("rvk_closure", tables)
        self.assertIn("concept_node", tables)
        self.assertIn("term_node", tables)
        self.assertIn("rvk_text_fts", tables)
        self.assertEqual(meta["schema_version"], self.index.SCHEMA_VERSION)
        self.assertEqual(meta["observed_max_ancestor_depth"], "17")

    def test_build_query_seeds_prioritizes_gnd_and_anchor_evidence(self):
        llm_analysis = LlmKeywordAnalysis(
            task_name="keywords",
            model_used="test",
            provider_used="test",
            prompt_template="",
            filled_prompt="",
            temperature=0.0,
            seed=0,
            response_full_text=(
                "<missing_list>Cadmiumbelastung; Bodenbelastung</missing_list>"
                "<keyword_chains>Cadmium, Boden -> Umweltproblem</keyword_chains>"
            ),
            analyse_text="Cadmium in belasteten Boeden",
        )

        seeds = self.index._build_query_seeds(
            keyword_entries=[
                {"keyword": "Cadmium (GND-ID: 4029921-1)", "gnd_id": "4029921-1"},
                {"keyword": "Boeden (GND-ID: 4007394-4)", "gnd_id": "4007394-4"},
            ],
            rvk_anchor_keywords=["Cadmium (GND-ID: 4029921-1)", "Boeden (GND-ID: 4007394-4)"],
            selected_dk_codes=["DK 631.4", "DK 628.5"],
            llm_analysis=llm_analysis,
        )

        self.assertTrue(seeds)
        self.assertEqual(seeds[0].seed_type, "gnd")
        self.assertIn("cadmium", {seed.value.casefold() for seed in seeds})
        self.assertIn("dk:631", {seed.value.casefold() for seed in seeds})

    def test_score_candidates_prefers_joint_authoritative_specific_candidate(self):
        seeds = [
            self.index._build_query_seeds(
                keyword_entries=[{"keyword": "Cadmium", "gnd_id": "4029921-1"}],
                rvk_anchor_keywords=["Cadmium", "Boden"],
                selected_dk_codes=[],
                llm_analysis=None,
            )
        ][0]
        seeds = list(seeds)

        broad = GraphCandidate(
            notation="RF 34000",
            label="Allgemeine Wirtschaftsgeografie",
            branch_family="R",
            ancestor_path="Geografie > Wirtschaftsgeografie",
            depth_from_root=3,
            evidence=[
                GraphEvidence("Cadmium", "gnd", "direct_concept", 0.8, ["Cadmium", "RF 34000"]),
            ],
        )
        specific = GraphCandidate(
            notation="RF 34699",
            label="Wasserkraft, Sonnenenergie, Windenergie",
            branch_family="R",
            ancestor_path="Geografie > ... > Nutzbare Kraftquellen",
            depth_from_root=11,
            evidence=[
                GraphEvidence("Cadmium", "gnd", "direct_concept", 1.0, ["Cadmium", "RF 34699"]),
                GraphEvidence("Boden", "anchor", "term", 0.9, ["Boden", "RF 34699"]),
            ],
        )

        scored = self.index._score_candidates(
            seeds=seeds,
            candidates=[broad, specific],
            original_abstract="Cadmium in Boeden und Umweltkontext",
            selected_dk_codes=[],
        )

        self.assertEqual(scored[0].notation, "RF 34699")
        self.assertGreater(scored[0].score, scored[1].score)

    def test_retrieve_candidates_returns_graph_based_keyword_results(self):
        self._write_mock_graph_index()

        results = self.index.retrieve_candidates(
            keyword_entries=[
                {"keyword": "Cadmium", "gnd_id": "4029921-1"},
                {"keyword": "Boeden", "gnd_id": "4007394-4"},
            ],
            original_abstract="Cadmiumbelastung in Boeden und Umweltkontext",
            rvk_anchor_keywords=["Cadmium", "Boeden"],
        )

        self.assertTrue(results)
        first_keyword = results[0]
        self.assertEqual(first_keyword["source"], "rvk_graph")
        first_classification = first_keyword["classifications"][0]
        self.assertEqual(first_classification["source"], "rvk_graph")
        self.assertTrue(first_classification["graph_evidence"])
        all_classifications = [
            classification
            for keyword_result in results
            for classification in keyword_result.get("classifications", [])
        ]
        all_notations = {classification["dk"] for classification in all_classifications}
        self.assertIn("RF 34699", all_notations)
        self.assertIn("RF 34000", all_notations)
        self.assertTrue(
            any(
                evidence.get("match_type") == "ancestor"
                for classification in all_classifications
                for evidence in classification.get("graph_evidence", [])
            )
        )
        self.assertTrue(
            any(
                evidence.get("match_type") == "sibling"
                for classification in all_classifications
                for evidence in classification.get("graph_evidence", [])
            )
        )


if __name__ == "__main__":
    unittest.main()

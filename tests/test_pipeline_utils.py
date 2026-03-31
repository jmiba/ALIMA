import logging
import os
import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_models import (
    AbstractData,
    AnalysisResult,
    LlmKeywordAnalysis,
    PromptConfigData,
    TaskState,
)

try:
    from src.utils.pipeline_utils import PipelineStepExecutor, export_analysis_state_to_file
    PIPELINE_UTILS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    PipelineStepExecutor = None
    export_analysis_state_to_file = None
    PIPELINE_UTILS_IMPORT_ERROR = exc


logging.disable(logging.CRITICAL)


@unittest.skipIf(
    PIPELINE_UTILS_IMPORT_ERROR is not None,
    f"Pipeline utilities dependencies unavailable: {PIPELINE_UTILS_IMPORT_ERROR}",
)
class TestPipelineStepExecutor(unittest.TestCase):
    def setUp(self):
        self.mock_alima_manager = Mock()
        self.mock_cache_manager = Mock()
        self.mock_logger = Mock()
        self.executor = PipelineStepExecutor(
            alima_manager=self.mock_alima_manager,
            cache_manager=self.mock_cache_manager,
            logger=self.mock_logger,
        )

    def _make_task_state(self, response_text: str, status: str = "completed") -> TaskState:
        return TaskState(
            abstract_data=AbstractData(abstract="Abstract", keywords=""),
            analysis_result=AnalysisResult(
                full_text=response_text,
                matched_keywords={},
                gnd_systematic="",
            ),
            prompt_config=PromptConfigData(
                prompt="Prompt",
                system="System",
                temp=0.0,
                p_value=1.0,
                models=["test-model"],
                seed=None,
                output_format="xml",
            ),
            status=status,
            task_name="dk_classification",
            model_used="test-model",
            provider_used="test-provider",
        )

    @staticmethod
    def _candidate_pool():
        return [
            {
                "dk": "720",
                "classification_type": "DK",
                "type": "DK",
                "count": 80,
                "matched_keywords": ["Architektur", "Klassizismus"],
                "titles": ["Titel A", "Titel B"],
            },
            {
                "dk": "56.63",
                "classification_type": "DK",
                "type": "DK",
                "count": 28,
                "matched_keywords": ["Profanarchitektur"],
                "titles": ["Titel C"],
            },
            {
                "dk": "LI 99999",
                "classification_type": "RVK",
                "type": "RVK",
                "count": 29,
                "matched_keywords": ["Klassizismus", "Architekturzeichnung"],
                "titles": [],
                "label": "Klassizistische Architektur",
                "ancestor_path": "Kunst > Architekturgeschichte",
                "branch_family": "L",
                "rvk_validation_status": "standard",
                "source": "rvk_graph",
            },
            {
                "dk": "LH 60320",
                "classification_type": "RVK",
                "type": "RVK",
                "count": 26,
                "matched_keywords": ["Profanarchitektur", "Architektur"],
                "titles": [],
                "label": "Profanarchitektur",
                "ancestor_path": "Kunst > Baukunst",
                "branch_family": "L",
                "rvk_validation_status": "standard",
                "source": "catalog",
            },
        ]

    def test_execute_initial_keyword_extraction(self):
        mock_task_state = TaskState(
            abstract_data=AbstractData(abstract="This is a test abstract.", keywords=""),
            analysis_result=AnalysisResult(
                full_text="<keywords>Machine Learning, AI</keywords><class>004</class>",
                matched_keywords={},
                gnd_systematic="",
            ),
            prompt_config=PromptConfigData(
                prompt="Test prompt",
                system="System prompt",
                temp=0.0,
                p_value=1.0,
                models=["test-model"],
                seed=None,
            ),
            status="completed",
            task_name="initialisation",
            model_used="test-model",
            provider_used="test-provider",
        )
        self.mock_alima_manager.analyze_abstract.return_value = mock_task_state

        keywords, gnd_classes, llm_analysis, llm_title = self.executor.execute_initial_keyword_extraction(
            abstract_text="This is a test abstract.",
            model="test-model",
            provider="test-provider",
            task="initialisation",
        )

        self.mock_alima_manager.analyze_abstract.assert_called_once()
        _, call_kwargs = self.mock_alima_manager.analyze_abstract.call_args
        self.assertEqual(call_kwargs["task"], "initialisation")
        self.assertEqual(call_kwargs["model"], "test-model")
        self.assertEqual(call_kwargs["provider"], "test-provider")
        self.assertEqual(keywords, ["Machine Learning", "AI"])
        self.assertEqual(gnd_classes, ["004"])
        self.assertIsInstance(llm_analysis, LlmKeywordAnalysis)
        self.assertIsNone(llm_title)

    def test_create_complete_analysis_state_converts_search_results(self):
        initial_analysis = Mock(spec=LlmKeywordAnalysis)
        final_analysis = Mock(spec=LlmKeywordAnalysis)

        state = self.executor.create_complete_analysis_state(
            original_abstract="Test abstract for full state creation.",
            initial_keywords=["initial", "keywords"],
            initial_gnd_classes=["001"],
            search_results={
                "term1": {"kw1": {"gndid": {"1"}}},
                "term2": {"kw2": {"gndid": {"2"}}},
            },
            initial_llm_analysis=initial_analysis,
            final_llm_analysis=final_analysis,
            suggesters_used=["lobid"],
        )

        self.assertEqual(state.original_abstract, "Test abstract for full state creation.")
        self.assertEqual(state.initial_keywords, ["initial", "keywords"])
        self.assertEqual(state.initial_gnd_classes, ["001"])
        self.assertEqual(state.search_suggesters_used, ["lobid"])
        self.assertEqual(len(state.search_results), 2)
        self.assertEqual(state.search_results[0].search_term, "term1")
        self.assertEqual(state.search_results[1].search_term, "term2")

    def test_execute_final_keyword_analysis_uses_batch_gnd_lookup(self):
        self.mock_cache_manager.get_gnd_facts_batch.return_value = {
            "4061694-5": SimpleNamespace(title="Umweltverschmutzung", synonyms=""),
            "4009274-4": SimpleNamespace(title="Cadmium", synonyms=""),
        }
        self.mock_alima_manager.analyze_abstract.return_value = TaskState(
            abstract_data=AbstractData(abstract="Abstract", keywords=""),
            analysis_result=AnalysisResult(
                full_text="<keywords>Umweltverschmutzung (GND-ID: 4061694-5), Cadmium (GND-ID: 4009274-4)</keywords><class>21.4</class>",
                matched_keywords={},
                gnd_systematic="",
            ),
            prompt_config=PromptConfigData(
                prompt="Prompt",
                system="System",
                temp=0.0,
                p_value=1.0,
                models=["final-model"],
                seed=None,
            ),
            status="completed",
            task_name="keywords",
            model_used="final-model",
            provider_used="final-provider",
        )

        final_keywords, gnd_classes, llm_analysis = self.executor.execute_final_keyword_analysis(
            original_abstract="Abstract",
            search_results={
                "environmental pollution": {
                    "Umweltverschmutzung": {"count": 50, "gndid": {"4061694-5"}},
                    "Cadmium": {"count": 20, "gndid": {"4009274-4"}},
                }
            },
            model="final-model",
            provider="final-provider",
            task="keywords",
            keyword_chunking_threshold=999,
        )

        self.mock_cache_manager.get_gnd_facts_batch.assert_called_once()
        self.assertEqual(gnd_classes, ["21.4"])
        self.assertIsInstance(llm_analysis, LlmKeywordAnalysis)
        self.assertTrue(any("4061694-5" in kw for kw in final_keywords))

    def test_export_analysis_state_to_file_uses_wrapped_schema(self):
        analysis_state = SimpleNamespace(
            original_abstract="Abstract",
            working_title="Title",
            initial_keywords=["one"],
            dk_classifications=["RVK QZ 123"],
            rvk_provenance={"rvk_api": 1},
            search_results=[],
            dk_search_results=[],
            dk_search_results_flattened=[],
            dk_statistics=None,
            search_suggesters_used=["lobid"],
            initial_gnd_classes=["001"],
            initial_llm_call_details=None,
            final_llm_analysis=SimpleNamespace(
                response_full_text="final response",
                provider_used="provider-b",
                model_used="model-b",
                extracted_keywords=[],
                extracted_gnd_keywords=["two"],
                extracted_gnd_classes=["123"],
                token_count=34,
                verification={"stats": {"verified_count": 1}},
            ),
            timestamp="2026-03-29T14:30:00",
        )

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            export_analysis_state_to_file(analysis_state, path)
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertIn("results", payload)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["results"]["original_abstract"], "Abstract")
            self.assertEqual(payload["results"]["final_keywords"], ["two"])
            self.assertEqual(payload["results"]["classifications"][0]["system"], "RVK")
        finally:
            os.unlink(path)

    def test_execute_dk_classification_keeps_valid_llm_result(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._make_task_state(
            "<final_list>DK 720 | RVK LI 99999</final_list>"
        )
        self.executor._select_final_dk_candidates = Mock(return_value=["DK 56.63"])
        self.executor._select_final_rvk_candidates = Mock(return_value=["RVK LH 60320"])
        self.executor._select_final_rvk_with_dk_context = Mock(return_value=[])
        self.executor._should_skip_dk_guided_rvk_second_pass = Mock(return_value=(False, [], ""))

        classifications, llm_analysis = self.executor.execute_dk_classification(
            original_abstract="Architektur und Klassizismus",
            dk_search_results=self._candidate_pool(),
            model="test-model",
            provider="test-provider",
        )

        self.assertEqual(classifications, ["DK 720", "RVK LI 99999"])
        self.assertEqual(llm_analysis.extracted_gnd_classes, ["DK 720", "RVK LI 99999"])

    def test_execute_dk_classification_uses_fallback_when_llm_returns_no_classes(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._make_task_state(
            "Analyse ohne verwertbare finale Klassen."
        )
        self.executor._select_final_dk_candidates = Mock(return_value=["DK 720"])
        self.executor._select_final_rvk_candidates = Mock(return_value=["RVK LI 99999"])
        self.executor._select_final_rvk_with_dk_context = Mock(return_value=[])
        self.executor._should_skip_dk_guided_rvk_second_pass = Mock(return_value=(False, [], ""))
        stream_callback = Mock()

        classifications, llm_analysis = self.executor.execute_dk_classification(
            original_abstract="Architektur und Klassizismus",
            dk_search_results=self._candidate_pool(),
            model="test-model",
            provider="test-provider",
            stream_callback=stream_callback,
        )

        self.assertEqual(classifications, ["DK 720", "RVK LI 99999"])
        self.assertEqual(llm_analysis.extracted_gnd_classes, ["DK 720", "RVK LI 99999"])
        streamed_text = "".join(call.args[0] for call in stream_callback.call_args_list)
        self.assertIn("LLM lieferte keine finalen Klassen", streamed_text)
        self.assertIn("Finale DK-Auswahl aus Fallback", streamed_text)
        self.assertIn("Finale RVK-Auswahl aus Fallback", streamed_text)

    def test_execute_dk_classification_tops_up_missing_rvk_from_fallback(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._make_task_state(
            "<final_list>DK 720</final_list>"
        )
        self.executor._select_final_dk_candidates = Mock(return_value=["DK 56.63"])
        self.executor._select_final_rvk_candidates = Mock(return_value=["RVK LI 99999"])
        self.executor._select_final_rvk_with_dk_context = Mock(return_value=[])
        self.executor._should_skip_dk_guided_rvk_second_pass = Mock(return_value=(False, [], ""))

        classifications, _ = self.executor.execute_dk_classification(
            original_abstract="Architektur und Klassizismus",
            dk_search_results=self._candidate_pool(),
            model="test-model",
            provider="test-provider",
        )

        self.assertEqual(classifications, ["DK 720", "RVK LI 99999"])

    def test_execute_dk_classification_tops_up_missing_dk_from_fallback(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._make_task_state(
            "<final_list>RVK LI 99999</final_list>"
        )
        self.executor._select_final_dk_candidates = Mock(return_value=["DK 720"])
        self.executor._select_final_rvk_candidates = Mock(return_value=["RVK LH 60320"])
        self.executor._select_final_rvk_with_dk_context = Mock(return_value=[])
        self.executor._should_skip_dk_guided_rvk_second_pass = Mock(return_value=(False, [], ""))

        classifications, _ = self.executor.execute_dk_classification(
            original_abstract="Architektur und Klassizismus",
            dk_search_results=self._candidate_pool(),
            model="test-model",
            provider="test-provider",
        )

        self.assertEqual(classifications, ["DK 720", "RVK LI 99999"])

    def test_execute_dk_classification_restores_filtered_rvk_from_fallback(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._make_task_state(
            "<final_list>DK 720 | RVK ZZ 12345</final_list>"
        )
        self.executor._select_final_dk_candidates = Mock(return_value=["DK 56.63"])
        self.executor._select_final_rvk_candidates = Mock(return_value=["RVK LI 99999"])
        self.executor._select_final_rvk_with_dk_context = Mock(return_value=[])
        self.executor._should_skip_dk_guided_rvk_second_pass = Mock(return_value=(False, [], ""))

        classifications, _ = self.executor.execute_dk_classification(
            original_abstract="Architektur und Klassizismus",
            dk_search_results=self._candidate_pool(),
            model="test-model",
            provider="test-provider",
        )

        self.assertEqual(classifications, ["DK 720", "RVK LI 99999"])

    def test_execute_dk_classification_does_not_forward_graph_flag_to_alima_manager(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._make_task_state(
            "<final_list>DK 720 | RVK LI 99999</final_list>"
        )
        self.executor._select_final_dk_candidates = Mock(return_value=["DK 56.63"])
        self.executor._select_final_rvk_candidates = Mock(return_value=["RVK LH 60320"])
        self.executor._select_final_rvk_with_dk_context = Mock(return_value=[])
        self.executor._should_skip_dk_guided_rvk_second_pass = Mock(return_value=(False, [], ""))

        classifications, _ = self.executor.execute_dk_classification(
            original_abstract="Architektur und Klassizismus",
            dk_search_results=self._candidate_pool(),
            model="test-model",
            provider="test-provider",
            use_rvk_graph_retrieval=True,
        )

        self.assertEqual(classifications, ["DK 720", "RVK LI 99999"])
        _, call_kwargs = self.mock_alima_manager.analyze_abstract.call_args
        self.assertNotIn("use_rvk_graph_retrieval", call_kwargs)

    def test_format_dk_results_for_prompt_includes_graph_evidence_for_rvk(self):
        from src.utils.pipeline_utils import PipelineResultFormatter

        prompt_text = PipelineResultFormatter.format_dk_results_for_prompt(
            [
                {
                    "dk": "LK 79000",
                    "classification_type": "RVK",
                    "count": 4,
                    "titles": [],
                    "matched_keywords": ["Architektur", "Klassizismus"],
                    "source": "rvk_graph",
                    "label": "Allgemeine geschichtliche Darstellungen, Handbücher",
                    "ancestor_path": "Kunstgeschichte > Deutschland > Architektur",
                    "register": ["Architektur", "Klassizismus"],
                    "rvk_validation_status": "standard",
                    "graph_joint_seed_count": 2,
                    "graph_parent_distance": 1,
                    "catalog_hit_count": 7,
                    "graph_evidence": [
                        {
                            "seed": "Architektur",
                            "seed_type": "gnd",
                            "match_type": "direct_concept",
                            "weight": 1.72,
                            "path": ["Architektur", "LK 79000"],
                        },
                        {
                            "seed": "Klassizismus",
                            "seed_type": "anchor",
                            "match_type": "ancestor",
                            "weight": 0.47,
                            "path": ["Klassizismus", "LK 95190", "LK 79000"],
                        },
                    ],
                }
            ]
        )

        self.assertIn("Quelle: RVK-Graph", prompt_text)
        self.assertIn("Graph-Seed-Abdeckung: 2", prompt_text)
        self.assertIn("Graph-Evidenz:", prompt_text)
        self.assertIn("Architektur: direkter GND-Treffer", prompt_text)
        self.assertIn("Katalog-Abdeckung: 7 Treffer", prompt_text)

    def test_enrich_graph_rvk_results_with_catalog_evidence_adds_catalog_hit_count(self):
        lookup = Mock()
        lookup.get_rsns_for_classification.side_effect = lambda classification: [101, 102, 102] if classification == "RVK LK 79000" else []
        lookup.get_titles_for_classification.return_value = [
            {"title": "Catalog Entry for RSN 101", "classifications": ["RVK LK 79000"]},
            {"title": "Catalog Entry for RSN 102", "classifications": ["RVK LK 79000"]},
        ]

        keyword_results = [
            {
                "keyword": "Architektur",
                "classifications": [
                    {
                        "dk": "LK 79000",
                        "classification_type": "RVK",
                        "source": "rvk_graph",
                        "graph_joint_seed_count": 2,
                        "graph_evidence": [{"seed": "Architektur", "match_type": "direct_concept", "path": ["Architektur", "LK 79000"]}],
                    }
                ],
            }
        ]

        with patch("src.utils.classification_lookup_service.get_classification_lookup_service", return_value=lookup):
            enriched = self.executor._enrich_graph_rvk_results_with_catalog_evidence(keyword_results)

        classification = enriched[0]["classifications"][0]
        self.assertEqual(classification["catalog_hit_count"], 2)
        self.assertEqual(classification["catalog_evidence_source"], "classification_lookup")

    def test_enrich_graph_rvk_results_uses_catalog_cache_fallback_when_json_index_empty(self):
        lookup = Mock()
        lookup.get_rsns_for_classification.return_value = []
        lookup.get_titles_for_classification.return_value = []

        ukm = Mock()
        ukm.get_catalog_titles_for_classification.return_value = (
            [
                {"rsn": 101, "title": "Titel 1", "classifications": ["RVK LK 79000"]},
                {"rsn": 102, "title": "Titel 2", "classifications": ["RVK LK 79000"]},
            ],
            7,
        )

        keyword_results = [
            {
                "keyword": "Architektur",
                "classifications": [
                    {
                        "dk": "LK 79000",
                        "classification_type": "RVK",
                        "source": "rvk_graph",
                        "graph_joint_seed_count": 2,
                    }
                ],
            }
        ]

        with patch("src.utils.classification_lookup_service.get_classification_lookup_service", return_value=lookup), \
             patch("src.utils.pipeline_utils.UnifiedKnowledgeManager", return_value=ukm):
            enriched = self.executor._enrich_graph_rvk_results_with_catalog_evidence(keyword_results)

        classification = enriched[0]["classifications"][0]
        self.assertEqual(classification["catalog_hit_count"], 7)
        self.assertEqual(classification["catalog_evidence_source"], "catalog_cache")
        self.assertEqual(classification["catalog_titles"], ["Titel 1", "Titel 2"])

    def test_flatten_keyword_centric_results_preserves_catalog_hit_count_for_rvk(self):
        flattened = self.executor._flatten_keyword_centric_results(
            [
                {
                    "keyword": "Architektur",
                    "classifications": [
                        {
                            "dk": "LK 79000",
                            "classification_type": "RVK",
                            "type": "RVK",
                            "count": 2,
                            "titles": [],
                            "matched_keywords": ["Architektur"],
                            "source": "rvk_graph",
                            "catalog_hit_count": 7,
                            "catalog_titles": ["Titel 1", "Titel 2"],
                        }
                    ],
                }
            ]
        )

        self.assertEqual(flattened[0]["catalog_hit_count"], 7)
        self.assertEqual(flattened[0]["catalog_titles"], ["Titel 1", "Titel 2"])


if __name__ == "__main__":
    unittest.main()

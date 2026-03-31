"""RVK graph index for graph-guided RVK retrieval."""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from ...core.processing_utils import (
    extract_analyse_text_from_response,
    extract_keyword_chains_from_response,
    extract_missing_concepts_from_response,
)

if TYPE_CHECKING:
    from ...core.data_models import LlmKeywordAnalysis


logger = logging.getLogger("rvk_graph_index")


@dataclass(frozen=True)
class GraphSeed:
    value: str
    seed_type: str
    weight: float = 1.0
    gnd_id: str = ""
    source_keyword: str = ""


@dataclass
class GraphEvidence:
    seed: str
    seed_type: str
    match_type: str
    weight: float
    path: List[str] = field(default_factory=list)


@dataclass
class GraphCandidate:
    notation: str
    label: str
    branch_family: str
    ancestor_path: str
    depth_from_root: int
    score: float = 0.0
    joint_seed_count: int = 0
    parent_distance: int = 0
    matched_keywords: List[str] = field(default_factory=list)
    register: List[str] = field(default_factory=list)
    evidence: List[GraphEvidence] = field(default_factory=list)

    def to_pipeline_dict(self) -> Dict[str, Any]:
        return {
            "dk": self.notation,
            "type": "RVK",
            "classification_type": "RVK",
            "count": max(1, int(self.joint_seed_count or 0)),
            "titles": [],
            "matched_keywords": list(self.matched_keywords),
            "source": "rvk_graph",
            "label": self.label,
            "ancestor_path": self.ancestor_path,
            "register": list(self.register),
            "score": float(self.score),
            "branch_family": self.branch_family,
            "rvk_validation_status": "standard",
            "validation_message": "",
            "graph_depth": int(self.depth_from_root or 0),
            "graph_joint_seed_count": int(self.joint_seed_count or 0),
            "graph_parent_distance": int(self.parent_distance or 0),
            "graph_evidence": [
                {
                    "seed": item.seed,
                    "seed_type": item.seed_type,
                    "match_type": item.match_type,
                    "weight": float(item.weight),
                    "path": list(item.path),
                }
                for item in self.evidence
            ],
        }


class RvkGraphIndex:
    """
    Local SQLite-backed RVK graph retriever.

    The currently observed maximum ancestor depth in RVK dump `2026_1` is 17.
    That value is only used as the default traversal cap and is written to DB
    metadata as an empirical observation, not as a schema constraint.
    """

    DEFAULT_MAX_ANCESTOR_HOPS = 17
    DEFAULT_MAX_DESCENDANT_HOPS = 17
    SCHEMA_VERSION = "1"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        timeout: int = 20,
        db_name: str = "rvk_graph.sqlite3",
    ) -> None:
        self.timeout = timeout
        self.data_dir = data_dir or self._default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / db_name

    @staticmethod
    def _default_data_dir() -> Path:
        from ..config_manager import ConfigManager

        config_manager = ConfigManager()
        return config_manager.config_file.parent / "rvk"

    @staticmethod
    def normalize_text(value: str) -> str:
        clean = str(value or "").strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean

    @staticmethod
    def normalize_keyword(value: str) -> str:
        clean = str(value or "").strip()
        if "(GND-ID:" in clean:
            clean = clean.split("(GND-ID:")[0].strip()
        return clean

    @staticmethod
    def normalize_seed_text(value: str) -> str:
        clean = RvkGraphIndex.normalize_keyword(value)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def normalize_gnd_id(value: str) -> str:
        clean = str(value or "").strip()
        match = re.search(r"\(DE-588\)\s*([0-9X-]+)", clean)
        if match:
            return match.group(1)
        match = re.search(r"([0-9X-]{6,})", clean)
        if match:
            return match.group(1)
        return clean

    @staticmethod
    def seed_type_rank(seed_type: str) -> int:
        return {
            "gnd": 0,
            "anchor": 1,
            "lexical": 2,
            "dk_branch": 3,
        }.get(str(seed_type or ""), 9)

    @staticmethod
    def split_reason_into_seed_phrases(text: str, max_phrases: int = 4) -> List[str]:
        parts = re.split(r"[;,\n]|(?:\bund\b)|(?:\boder\b)", str(text or ""), flags=re.IGNORECASE)
        cleaned: List[str] = []
        seen: Set[str] = set()
        for part in parts:
            normalized = RvkGraphIndex.normalize_seed_text(part)
            if len(normalized) < 3:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            cleaned.append(normalized)
            seen.add(key)
            if len(cleaned) >= max_phrases:
                break
        return cleaned

    @classmethod
    def infer_branch_hints_from_dk_codes(cls, selected_dk_codes: Sequence[str]) -> List[str]:
        hints: List[str] = []
        seen: Set[str] = set()

        for code in selected_dk_codes or []:
            clean = str(code or "").strip().upper()
            if not clean:
                continue
            if clean.startswith("RVK "):
                continue
            if clean.startswith("DK "):
                clean = clean[3:].strip()
            major = clean.split(".", 1)[0]
            if not major:
                continue
            hint = f"dk:{major}"
            if hint not in seen:
                hints.append(hint)
                seen.add(hint)

        return hints

    @classmethod
    def infer_branch_hints_from_candidate_text(
        cls,
        label: str,
        ancestor_path: str,
        abstract_text: str,
    ) -> List[str]:
        text = " ".join(
            part for part in (cls.normalize_text(label), cls.normalize_text(ancestor_path)) if part
        ).casefold()
        abstract = cls.normalize_text(abstract_text).casefold()

        branch_map = {
            "geografie": ["geo", "anthropogeografie", "wirtschaftsgeografie"],
            "wirtschaft": ["economics", "industrie", "handel"],
            "chemie": ["chem", "chemisch", "molekular"],
            "geschichte": ["history", "histor", "epoche"],
            "recht": ["law", "juristisch", "rechtlich"],
            "medizin": ["medizin", "klinisch", "gesundheit"],
            "technik": ["technik", "ingenieur", "verfahren"],
            "politik": ["politik", "staat", "regierung"],
        }

        hints: List[str] = []
        for key, markers in branch_map.items():
            if key in text or any(marker in abstract for marker in markers):
                hints.append(key)
        return hints

    @classmethod
    def branch_alignment_score(cls, branch_family: str, branch_hints: Sequence[str]) -> float:
        if not branch_family or not branch_hints:
            return 0.0

        family = str(branch_family or "").strip().upper()
        hints = {str(item or "").strip().casefold() for item in branch_hints if str(item or "").strip()}
        if not hints:
            return 0.0

        if family.casefold() in hints:
            return 1.0

        family_to_hint = {
            "R": {"geo"},
            "Z": {"economics", "politik", "technik"},
            "Q": {"medizin"},
            "V": {"chem"},
            "N": {"geschichte"},
            "P": {"recht", "law"},
        }
        expected = family_to_hint.get(family, set())
        if hints.intersection(expected):
            return 0.75
        return 0.0

    @staticmethod
    def specificity_score(depth: int, branch_family: str) -> float:
        # Reward middle/deeper nodes without forcing the deepest node in every branch.
        if depth <= 0:
            return 0.0
        if depth <= 2:
            return 0.15
        if depth <= 4:
            return 0.45
        if depth <= 8:
            return 0.85
        if depth <= 12:
            return 1.0
        return 0.92

    @staticmethod
    def _extract_candidate_seed_support(candidate: GraphCandidate) -> Dict[str, float]:
        support: Dict[str, float] = {}
        for ev in candidate.evidence or []:
            current = support.get(ev.seed, 0.0)
            support[ev.seed] = max(current, float(ev.weight))
        return support

    @classmethod
    def has_more_specific_child_with_similar_support(
        cls,
        candidate: GraphCandidate,
        all_candidates: Sequence[GraphCandidate],
        min_overlap_ratio: float = 0.75,
    ) -> bool:
        candidate_support = cls._extract_candidate_seed_support(candidate)
        if not candidate_support:
            return False

        candidate_seeds = set(candidate_support.keys())
        candidate_branch = str(candidate.branch_family or "")
        candidate_depth = int(candidate.depth_from_root or 0)

        for other in all_candidates:
            if other.notation == candidate.notation:
                continue
            if str(other.branch_family or "") != candidate_branch:
                continue
            if int(other.depth_from_root or 0) <= candidate_depth:
                continue

            other_support = cls._extract_candidate_seed_support(other)
            if not other_support:
                continue
            overlap = candidate_seeds.intersection(other_support.keys())
            if not overlap:
                continue
            if len(overlap) / max(1, len(candidate_seeds)) >= min_overlap_ratio:
                if float(sum(other_support.values())) >= float(sum(candidate_support.values())) * 0.9:
                    return True
        return False

    @classmethod
    def boost_seed_agreement(cls, seeds: Sequence[GraphSeed]) -> List[GraphSeed]:
        if not seeds:
            return []

        grouped: Dict[str, List[GraphSeed]] = {}
        for seed in seeds:
            grouped.setdefault(seed.value.casefold(), []).append(seed)

        boosted: List[GraphSeed] = []
        for _, group in grouped.items():
            agreement_bonus = 0.12 * max(0, len(group) - 1)
            gnd_bonus = 0.15 if any(seed.seed_type == "gnd" for seed in group) else 0.0
            anchor_bonus = 0.08 if any(seed.seed_type == "anchor" for seed in group) else 0.0
            for seed in group:
                boosted.append(
                    GraphSeed(
                        value=seed.value,
                        seed_type=seed.seed_type,
                        weight=round(float(seed.weight) + agreement_bonus + gnd_bonus + anchor_bonus, 4),
                        gnd_id=seed.gnd_id,
                        source_keyword=seed.source_keyword,
                    )
                )
        return boosted

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _strip_namespace(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    @staticmethod
    def _branch_family(notation: str) -> str:
        match = re.search(r"[A-Z]", str(notation or ""))
        return match.group(0) if match else ""

    def _read_meta(self, conn: sqlite3.Connection) -> Dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key, value FROM meta").fetchall()
        }

    @staticmethod
    def _node_count(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COUNT(*) AS count FROM rvk_node").fetchone()
        return int(row["count"] if row and row["count"] is not None else 0)

    def _clear_graph_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM rvk_text_fts")
        for table_name in [
            "branch_profile",
            "term_rvk_edge",
            "term_node",
            "concept_rvk_edge",
            "concept_node",
            "rvk_closure",
            "rvk_edge",
            "rvk_node",
        ]:
            conn.execute(f"DELETE FROM {table_name}")
        conn.commit()

    def _resolve_dump_metadata(self, progress_callback=None) -> Tuple[Optional[Path], str]:
        from .rvk_marc_index import RvkMarcIndex

        index = RvkMarcIndex(data_dir=self.data_dir, timeout=self.timeout)
        if not index.ensure_index(progress_callback=progress_callback):
            return None, ""
        meta = index._read_meta()
        dump_path = Path(meta["dump_path"]) if meta.get("dump_path") else None
        if dump_path and dump_path.exists():
            return dump_path, str(meta.get("release", "") or "")
        return None, str(meta.get("release", "") or "")

    @staticmethod
    def _normalize_term_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    def _parse_dump_record(
        self,
        record: ET.Element,
    ) -> Tuple[str, str, List[str], List[Tuple[str, str, str]]]:
        notation = ""
        label = ""
        hierarchy_labels: List[str] = []
        gnd_rows: List[Tuple[str, str, str]] = []

        for datafield in record:
            if self._strip_namespace(datafield.tag) != "datafield":
                continue
            tag = datafield.attrib.get("tag", "")

            if tag == "153":
                for subfield in datafield:
                    if self._strip_namespace(subfield.tag) != "subfield":
                        continue
                    code = subfield.attrib.get("code", "")
                    value = self.normalize_text(subfield.text or "")
                    if code == "a" and not notation:
                        notation = value
                    elif code == "j" and not label:
                        label = value
                    elif code == "h" and value:
                        hierarchy_labels.append(value)
                continue

            if tag not in {"700", "710", "711", "730", "750", "751"}:
                continue

            gnd_id = ""
            register_term = ""
            source = ""
            for subfield in datafield:
                if self._strip_namespace(subfield.tag) != "subfield":
                    continue
                code = subfield.attrib.get("code", "")
                value = self.normalize_text(subfield.text or "")
                if code == "0":
                    gnd_id = self.normalize_gnd_id(value)
                elif code == "a":
                    register_term = value
                elif code == "2":
                    source = value.lower()
            if gnd_id and register_term and source == "gnd":
                gnd_rows.append((gnd_id, register_term, tag))

        return notation, label, hierarchy_labels, gnd_rows

    def ensure_graph(self, force_rebuild: bool = False, progress_callback=None) -> bool:
        created = force_rebuild or not self.db_path.exists()
        conn = self._connect()
        rebuild_reasons: List[str] = []
        try:
            self._create_schema(conn)
            meta = self._read_meta(conn)
            node_count = self._node_count(conn)
            dump_path, release = self._resolve_dump_metadata(progress_callback=progress_callback)
            current_release = str(meta.get("release", "") or "")
            schema_version = str(meta.get("schema_version", "") or "")
            if force_rebuild:
                rebuild_reasons.append("erzwungener Neuaufbau")
            if node_count == 0:
                rebuild_reasons.append("kein vorhandener Graphinhalt")
            if schema_version != self.SCHEMA_VERSION:
                rebuild_reasons.append(
                    f"Schema-Version {schema_version or 'leer'} != {self.SCHEMA_VERSION}"
                )
            if release and release != current_release:
                rebuild_reasons.append(
                    f"Release-Wechsel {current_release or 'unbekannt'} → {release}"
                )
            needs_build = bool(rebuild_reasons)
        finally:
            conn.close()

        if needs_build and dump_path and dump_path.exists():
            if progress_callback:
                progress_callback(
                    "  ↻ RVK-Graphindex wird neu aufgebaut: "
                    + ", ".join(rebuild_reasons)
                    + "\n"
                )
            self._build_graph_from_dump(dump_path, release or "unknown", progress_callback=progress_callback)
        elif needs_build:
            conn = self._connect()
            try:
                meta_rows = [
                    ("schema_version", self.SCHEMA_VERSION),
                    ("initialized_at", datetime.now(UTC).isoformat()),
                    ("observed_max_ancestor_depth", str(self.DEFAULT_MAX_ANCESTOR_HOPS)),
                    ("default_max_ancestor_hops", str(self.DEFAULT_MAX_ANCESTOR_HOPS)),
                    ("default_max_descendant_hops", str(self.DEFAULT_MAX_DESCENDANT_HOPS)),
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    meta_rows,
                )
                conn.commit()
            finally:
                conn.close()
            if progress_callback:
                progress_callback(
                    "  ⚠️ RVK-Graph konnte nicht aus Dump gebaut werden: "
                    + ", ".join(rebuild_reasons)
                    + "\n"
                )

        if progress_callback:
            if needs_build and dump_path and dump_path.exists():
                progress_callback(
                    f"  ✅ RVK-Graphindex bereit ({release or 'unbekannt'}, {node_count or 'neu'} → lokal aktualisiert)\n"
                )
            elif created:
                progress_callback("  ✅ RVK-Graphschema initialisiert\n")
            else:
                progress_callback(
                    f"  ℹ️ RVK-Graph wiederverwendet ({current_release or 'unbekannt'}, {node_count} Knoten)\n"
                )
        return True

    def retrieve_candidates(
        self,
        keyword_entries: List[Dict[str, str]],
        original_abstract: str,
        rvk_anchor_keywords: Optional[List[str]] = None,
        selected_dk_codes: Optional[List[str]] = None,
        llm_analysis: Optional["LlmKeywordAnalysis"] = None,
        max_results_per_seed: int = 8,
        max_graph_candidates: int = 48,
        max_ancestor_hops: int = DEFAULT_MAX_ANCESTOR_HOPS,
        max_descendant_hops: int = DEFAULT_MAX_DESCENDANT_HOPS,
        max_sibling_expansions: int = 6,
        min_context_gain: float = 0.12,
        min_branch_fit: float = 0.30,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        self.ensure_graph(progress_callback=progress_callback)

        seeds = self._build_query_seeds(
            keyword_entries=keyword_entries,
            rvk_anchor_keywords=rvk_anchor_keywords,
            selected_dk_codes=selected_dk_codes,
            llm_analysis=llm_analysis,
        )
        if progress_callback:
            progress_callback(f"  ↳ RVK-Graph: {len(seeds)} Seeds vorbereitet\n")
        if not seeds:
            return []

        direct_hits = self._fetch_direct_seed_hits(seeds, limit=max_results_per_seed)
        expanded_candidates = self._expand_neighborhood(
            seeds=seeds,
            direct_hits=direct_hits,
            max_ancestor_hops=max_ancestor_hops,
            max_descendant_hops=max_descendant_hops,
            max_sibling_expansions=max_sibling_expansions,
            min_context_gain=min_context_gain,
            min_branch_fit=min_branch_fit,
        )
        if not expanded_candidates:
            if progress_callback:
                progress_callback("  ⚠️ RVK-Graph: keine authority-backed Kandidaten gefunden\n")
            return []

        scored_candidates = self._score_candidates(
            seeds=seeds,
            candidates=expanded_candidates,
            original_abstract=original_abstract,
            selected_dk_codes=selected_dk_codes or [],
        )
        selected_candidates = self._select_diverse_shortlist(
            scored_candidates,
            limit=max_graph_candidates,
        )
        results = self._format_keyword_centric_results(selected_candidates, keyword_entries)
        if progress_callback:
            total = sum(len(item.get("classifications", [])) for item in results)
            progress_callback(
                f"  ✅ RVK-Graph: {total} graph-basierte Kandidaten für {len(results)} Keywords\n"
            )
        return results

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rvk_node (
                notation TEXT PRIMARY KEY,
                label TEXT NOT NULL DEFAULT '',
                branch_family TEXT NOT NULL DEFAULT '',
                ancestor_path TEXT NOT NULL DEFAULT '',
                depth_from_root INTEGER NOT NULL DEFAULT 0 CHECK (depth_from_root >= 0),
                normalized_depth REAL NOT NULL DEFAULT 0.0,
                semantic_text TEXT NOT NULL DEFAULT '',
                is_leaf INTEGER NOT NULL DEFAULT 0 CHECK (is_leaf IN (0, 1))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rvk_edge (
                parent_notation TEXT NOT NULL,
                child_notation TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'parent_child',
                PRIMARY KEY (parent_notation, child_notation),
                FOREIGN KEY (parent_notation) REFERENCES rvk_node(notation) ON DELETE CASCADE,
                FOREIGN KEY (child_notation) REFERENCES rvk_node(notation) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_rvk_edge_parent ON rvk_edge(parent_notation)",
            "CREATE INDEX IF NOT EXISTS idx_rvk_edge_child ON rvk_edge(child_notation)",
            """
            CREATE TABLE IF NOT EXISTS rvk_closure (
                ancestor_notation TEXT NOT NULL,
                descendant_notation TEXT NOT NULL,
                distance INTEGER NOT NULL CHECK (distance >= 0),
                PRIMARY KEY (ancestor_notation, descendant_notation),
                FOREIGN KEY (ancestor_notation) REFERENCES rvk_node(notation) ON DELETE CASCADE,
                FOREIGN KEY (descendant_notation) REFERENCES rvk_node(notation) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_rvk_closure_ancestor ON rvk_closure(ancestor_notation, distance)",
            "CREATE INDEX IF NOT EXISTS idx_rvk_closure_descendant ON rvk_closure(descendant_notation, distance)",
            """
            CREATE TABLE IF NOT EXISTS concept_node (
                gnd_id TEXT PRIMARY KEY,
                preferred_label TEXT NOT NULL DEFAULT '',
                normalized_label TEXT NOT NULL DEFAULT ''
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS concept_rvk_edge (
                gnd_id TEXT NOT NULL,
                notation TEXT NOT NULL,
                field_tag TEXT NOT NULL DEFAULT '',
                register_term TEXT NOT NULL DEFAULT '',
                direct_weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (gnd_id, notation, field_tag, register_term),
                FOREIGN KEY (gnd_id) REFERENCES concept_node(gnd_id) ON DELETE CASCADE,
                FOREIGN KEY (notation) REFERENCES rvk_node(notation) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_concept_rvk_gnd ON concept_rvk_edge(gnd_id, direct_weight DESC)",
            "CREATE INDEX IF NOT EXISTS idx_concept_rvk_notation ON concept_rvk_edge(notation)",
            """
            CREATE TABLE IF NOT EXISTS term_node (
                term_norm TEXT PRIMARY KEY,
                display_term TEXT NOT NULL DEFAULT '',
                term_kind TEXT NOT NULL DEFAULT 'register'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS term_rvk_edge (
                term_norm TEXT NOT NULL,
                notation TEXT NOT NULL,
                source_field TEXT NOT NULL DEFAULT 'register',
                weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (term_norm, notation, source_field),
                FOREIGN KEY (term_norm) REFERENCES term_node(term_norm) ON DELETE CASCADE,
                FOREIGN KEY (notation) REFERENCES rvk_node(notation) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_term_rvk_term ON term_rvk_edge(term_norm, weight DESC)",
            "CREATE INDEX IF NOT EXISTS idx_term_rvk_notation ON term_rvk_edge(notation)",
            """
            CREATE TABLE IF NOT EXISTS branch_profile (
                branch_family TEXT PRIMARY KEY,
                dominant_terms_json TEXT NOT NULL DEFAULT '[]',
                dominant_concepts_json TEXT NOT NULL DEFAULT '[]'
            )
            """,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS rvk_text_fts
            USING fts5(
                notation UNINDEXED,
                label,
                ancestor_path,
                semantic_text,
                content=''
            )
            """,
        ]
        for statement in statements:
            conn.execute(statement)
        conn.commit()

    def _build_graph_from_dump(self, dump_path: Path, release: str, progress_callback=None) -> None:
        fd, temp_path = tempfile.mkstemp(prefix="rvk_graph_", suffix=".sqlite3", dir=str(self.data_dir))
        os.close(fd)
        temp_db = Path(temp_path)

        node_rows: Dict[str, Dict[str, Any]] = {}
        full_path_by_notation: Dict[str, Tuple[str, ...]] = {}
        path_to_notations: Dict[Tuple[str, ...], Set[str]] = {}
        concept_labels: Dict[str, str] = {}
        concept_edges: Set[Tuple[str, str, str, str, float]] = set()
        term_nodes: Dict[str, Tuple[str, str]] = {}
        term_edges: Set[Tuple[str, str, str, float]] = set()
        branch_term_counter: Dict[str, Counter] = {}
        branch_concept_counter: Dict[str, Counter] = {}
        records_processed = 0

        try:
            conn = sqlite3.connect(temp_db)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._create_schema(conn)

            with gzip.open(dump_path, "rb") as fh:
                for _, elem in ET.iterparse(fh, events=("end",)):
                    if self._strip_namespace(elem.tag) != "record":
                        continue

                    notation, label, hierarchy_labels, gnd_rows = self._parse_dump_record(elem)
                    records_processed += 1
                    if not notation or not label:
                        elem.clear()
                        continue

                    branch_family = self._branch_family(notation)
                    ancestor_path = " > ".join(hierarchy_labels)
                    full_path = tuple(hierarchy_labels + [label])
                    register_terms = sorted({register_term for _, register_term, _ in gnd_rows if register_term})
                    semantic_parts = [label, ancestor_path, " | ".join(register_terms)]
                    semantic_text = " | ".join(part for part in semantic_parts if part)

                    node_rows[notation] = {
                        "notation": notation,
                        "label": label,
                        "branch_family": branch_family,
                        "ancestor_path": ancestor_path,
                        "depth_from_root": len(hierarchy_labels),
                        "normalized_depth": round(
                            len(hierarchy_labels) / max(self.DEFAULT_MAX_ANCESTOR_HOPS, 1),
                            4,
                        ),
                        "semantic_text": semantic_text,
                        "is_leaf": 1,
                    }
                    full_path_by_notation[notation] = full_path
                    path_to_notations.setdefault(full_path, set()).add(notation)

                    label_key = self._normalize_term_key(label)
                    if label_key:
                        term_nodes[label_key] = (label, "label")
                        term_edges.add((label_key, notation, "label", 0.85))
                        branch_term_counter.setdefault(branch_family, Counter())[label] += 1

                    for gnd_id, register_term, field_tag in gnd_rows:
                        concept_labels.setdefault(gnd_id, register_term)
                        concept_edges.add((gnd_id, notation, field_tag, register_term, 1.0))
                        branch_concept_counter.setdefault(branch_family, Counter())[gnd_id] += 1
                        term_key = self._normalize_term_key(register_term)
                        if term_key:
                            term_nodes[term_key] = (register_term, "register")
                            term_edges.add((term_key, notation, "register", 1.0))
                            branch_term_counter.setdefault(branch_family, Counter())[register_term] += 1

                    if progress_callback and records_processed % 20000 == 0:
                        progress_callback(
                            f"  ↳ RVK-Graph: {records_processed} Datensätze verarbeitet\n"
                        )

                    elem.clear()

            unique_path_map = {
                path: next(iter(notations))
                for path, notations in path_to_notations.items()
                if len(notations) == 1
            }

            edge_rows: Set[Tuple[str, str, str]] = set()
            closure_distances: Dict[Tuple[str, str], int] = {}
            parent_children: Dict[str, Set[str]] = {}

            for notation, full_path in full_path_by_notation.items():
                closure_distances[(notation, notation)] = 0
                for prefix_len in range(1, len(full_path)):
                    ancestor_notation = unique_path_map.get(full_path[:prefix_len])
                    if not ancestor_notation:
                        continue
                    distance = len(full_path) - prefix_len
                    key = (ancestor_notation, notation)
                    existing_distance = closure_distances.get(key)
                    if existing_distance is None:
                        closure_distances[key] = distance
                    else:
                        closure_distances[key] = min(existing_distance, distance)
                    if prefix_len == len(full_path) - 1:
                        edge_rows.add((ancestor_notation, notation, "parent_child"))
                        parent_children.setdefault(ancestor_notation, set()).add(notation)

            for parent_notation, child_notations in parent_children.items():
                if parent_notation in node_rows and child_notations:
                    node_rows[parent_notation]["is_leaf"] = 0

            self._clear_graph_tables(conn)

            conn.executemany(
                """
                INSERT INTO rvk_node
                (notation, label, branch_family, ancestor_path, depth_from_root, normalized_depth, semantic_text, is_leaf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row["notation"],
                        row["label"],
                        row["branch_family"],
                        row["ancestor_path"],
                        row["depth_from_root"],
                        row["normalized_depth"],
                        row["semantic_text"],
                        row["is_leaf"],
                    )
                    for row in node_rows.values()
                ],
            )
            conn.executemany(
                "INSERT INTO rvk_edge (parent_notation, child_notation, edge_type) VALUES (?, ?, ?)",
                list(edge_rows),
            )
            conn.executemany(
                "INSERT INTO rvk_closure (ancestor_notation, descendant_notation, distance) VALUES (?, ?, ?)",
                [
                    (ancestor_notation, descendant_notation, distance)
                    for (ancestor_notation, descendant_notation), distance in closure_distances.items()
                ],
            )
            conn.executemany(
                "INSERT INTO concept_node (gnd_id, preferred_label, normalized_label) VALUES (?, ?, ?)",
                [
                    (gnd_id, label, self._normalize_term_key(label))
                    for gnd_id, label in concept_labels.items()
                ],
            )
            conn.executemany(
                """
                INSERT INTO concept_rvk_edge
                (gnd_id, notation, field_tag, register_term, direct_weight)
                VALUES (?, ?, ?, ?, ?)
                """,
                list(concept_edges),
            )
            conn.executemany(
                "INSERT INTO term_node (term_norm, display_term, term_kind) VALUES (?, ?, ?)",
                [
                    (term_norm, display_term, term_kind)
                    for term_norm, (display_term, term_kind) in term_nodes.items()
                ],
            )
            conn.executemany(
                """
                INSERT INTO term_rvk_edge
                (term_norm, notation, source_field, weight)
                VALUES (?, ?, ?, ?)
                """,
                list(term_edges),
            )
            conn.executemany(
                """
                INSERT INTO rvk_text_fts (notation, label, ancestor_path, semantic_text)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        row["notation"],
                        row["label"],
                        row["ancestor_path"],
                        row["semantic_text"],
                    )
                    for row in node_rows.values()
                ],
            )

            branch_profiles = []
            for branch_family, counter in branch_term_counter.items():
                branch_profiles.append(
                    (
                        branch_family,
                        json.dumps([term for term, _ in counter.most_common(20)], ensure_ascii=False),
                        json.dumps(
                            [gnd_id for gnd_id, _ in branch_concept_counter.get(branch_family, Counter()).most_common(20)],
                            ensure_ascii=False,
                        ),
                    )
                )
            if branch_profiles:
                conn.executemany(
                    """
                    INSERT INTO branch_profile
                    (branch_family, dominant_terms_json, dominant_concepts_json)
                    VALUES (?, ?, ?)
                    """,
                    branch_profiles,
                )

            conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", self.SCHEMA_VERSION),
                    ("release", release),
                    ("built_at", datetime.now(UTC).isoformat()),
                    ("dump_path", str(dump_path)),
                    ("records_processed", str(records_processed)),
                    ("node_count", str(len(node_rows))),
                    ("edge_count", str(len(edge_rows))),
                    ("closure_count", str(len(closure_distances))),
                    ("concept_edge_count", str(len(concept_edges))),
                    ("term_edge_count", str(len(term_edges))),
                    ("observed_max_ancestor_depth", str(self.DEFAULT_MAX_ANCESTOR_HOPS)),
                    ("default_max_ancestor_hops", str(self.DEFAULT_MAX_ANCESTOR_HOPS)),
                    ("default_max_descendant_hops", str(self.DEFAULT_MAX_DESCENDANT_HOPS)),
                ],
            )
            conn.commit()
            conn.close()
            temp_db.replace(self.db_path)
        finally:
            if temp_db.exists():
                temp_db.unlink(missing_ok=True)

    def _materialize_closure(self, conn: sqlite3.Connection) -> None:
        # Closure is materialized during dump import from the explicit hierarchy paths.
        conn.commit()

    @staticmethod
    def _parse_csv_values(value: Any) -> List[str]:
        seen: Set[str] = set()
        values: List[str] = []
        for item in str(value or "").split(","):
            clean = RvkGraphIndex.normalize_text(item)
            if not clean:
                continue
            key = clean.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(clean)
        return values

    def _query_node_metadata(
        self,
        conn: sqlite3.Connection,
        notations: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        notation_list = [self.normalize_text(item) for item in notations if self.normalize_text(item)]
        if not notation_list:
            return {}

        placeholders = ",".join("?" for _ in notation_list)
        rows = conn.execute(
            f"""
            SELECT
                n.notation,
                n.label,
                n.branch_family,
                n.ancestor_path,
                n.depth_from_root,
                n.normalized_depth,
                n.semantic_text,
                n.is_leaf,
                GROUP_CONCAT(DISTINCT tn.display_term) AS register_terms,
                GROUP_CONCAT(DISTINCT cr.gnd_id) AS gnd_ids
            FROM rvk_node n
            LEFT JOIN term_rvk_edge tr
                ON tr.notation = n.notation
            LEFT JOIN term_node tn
                ON tn.term_norm = tr.term_norm
            LEFT JOIN concept_rvk_edge cr
                ON cr.notation = n.notation
            WHERE n.notation IN ({placeholders})
            GROUP BY
                n.notation,
                n.label,
                n.branch_family,
                n.ancestor_path,
                n.depth_from_root,
                n.normalized_depth,
                n.semantic_text,
                n.is_leaf
            """,
            notation_list,
        ).fetchall()

        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            notation = self.normalize_text(row["notation"])
            result[notation] = {
                "notation": notation,
                "label": self.normalize_text(row["label"]),
                "branch_family": self.normalize_text(row["branch_family"]),
                "ancestor_path": self.normalize_text(row["ancestor_path"]),
                "depth_from_root": int(row["depth_from_root"] or 0),
                "normalized_depth": float(row["normalized_depth"] or 0.0),
                "semantic_text": self.normalize_text(row["semantic_text"]),
                "is_leaf": bool(row["is_leaf"]),
                "register_terms": self._parse_csv_values(row["register_terms"]),
                "gnd_ids": set(self._parse_csv_values(row["gnd_ids"])),
            }
        return result

    def _rank_seed_term_match(
        self,
        seed: GraphSeed,
        label: str,
        ancestor_path: str,
        register_terms: Sequence[str],
    ) -> int:
        seed_key = self._normalize_term_key(seed.value)
        if not seed_key:
            return 0

        label_key = self._normalize_term_key(label)
        ancestor_key = self._normalize_term_key(ancestor_path)
        register_keys = {self._normalize_term_key(term) for term in register_terms if self._normalize_term_key(term)}

        if seed_key in register_keys:
            return 4
        if seed_key == label_key:
            return 3
        if any(seed_key in item for item in register_keys):
            return 2
        if seed_key and (seed_key in label_key or seed_key in ancestor_key):
            return 1
        return 0

    def _build_local_node_evidence(
        self,
        node: Dict[str, Any],
        seeds: Sequence[GraphSeed],
    ) -> Tuple[List[GraphEvidence], List[str]]:
        evidence: List[GraphEvidence] = []
        matched_keywords: List[str] = []
        seen_keywords: Set[str] = set()

        gnd_ids = {self.normalize_gnd_id(item) for item in node.get("gnd_ids", set()) if item}
        label = self.normalize_text(node.get("label", ""))
        ancestor_path = self.normalize_text(node.get("ancestor_path", ""))
        register_terms = list(node.get("register_terms", []) or [])
        notation = self.normalize_text(node.get("notation", ""))

        for seed in seeds:
            if seed.seed_type == "gnd" and seed.gnd_id:
                if self.normalize_gnd_id(seed.gnd_id) not in gnd_ids:
                    continue
                evidence.append(
                    GraphEvidence(
                        seed=seed.value,
                        seed_type=seed.seed_type,
                        match_type="direct_concept",
                        weight=round(float(seed.weight) + 0.25, 4),
                        path=[seed.value, notation],
                    )
                )
                keyword = self.normalize_seed_text(seed.source_keyword or seed.value)
                if keyword and keyword.casefold() not in seen_keywords:
                    matched_keywords.append(keyword)
                    seen_keywords.add(keyword.casefold())
                continue

            if seed.seed_type not in {"anchor", "lexical"}:
                continue

            lexical_rank = self._rank_seed_term_match(seed, label, ancestor_path, register_terms)
            if lexical_rank <= 0:
                continue

            weight = float(seed.weight) + lexical_rank * 0.14
            if lexical_rank >= 3:
                weight += 0.08
            evidence.append(
                GraphEvidence(
                    seed=seed.value,
                    seed_type=seed.seed_type,
                    match_type="term",
                    weight=round(weight, 4),
                    path=[seed.value, notation],
                )
            )
            keyword = self.normalize_seed_text(seed.source_keyword or seed.value)
            if keyword and keyword.casefold() not in seen_keywords:
                matched_keywords.append(keyword)
                seen_keywords.add(keyword.casefold())

        return evidence, matched_keywords

    def _make_candidate_from_node(
        self,
        node: Dict[str, Any],
        evidence: Optional[Sequence[GraphEvidence]] = None,
        matched_keywords: Optional[Sequence[str]] = None,
        parent_distance: int = 0,
    ) -> GraphCandidate:
        return GraphCandidate(
            notation=self.normalize_text(node.get("notation", "")),
            label=self.normalize_text(node.get("label", "")),
            branch_family=self.normalize_text(node.get("branch_family", "")),
            ancestor_path=self.normalize_text(node.get("ancestor_path", "")),
            depth_from_root=int(node.get("depth_from_root") or 0),
            matched_keywords=list(matched_keywords or []),
            register=list(node.get("register_terms", []) or []),
            evidence=list(evidence or []),
            parent_distance=int(parent_distance or 0),
        )

    @staticmethod
    def _copy_candidate(candidate: GraphCandidate) -> GraphCandidate:
        return GraphCandidate(
            notation=candidate.notation,
            label=candidate.label,
            branch_family=candidate.branch_family,
            ancestor_path=candidate.ancestor_path,
            depth_from_root=int(candidate.depth_from_root or 0),
            score=float(candidate.score or 0.0),
            joint_seed_count=int(candidate.joint_seed_count or 0),
            parent_distance=int(candidate.parent_distance or 0),
            matched_keywords=list(candidate.matched_keywords or []),
            register=list(candidate.register or []),
            evidence=list(candidate.evidence or []),
        )

    def _merge_candidate(self, aggregated: Dict[str, GraphCandidate], candidate: GraphCandidate) -> None:
        existing = aggregated.get(candidate.notation)
        if existing is None:
            aggregated[candidate.notation] = self._copy_candidate(candidate)
            return

        if candidate.label and not existing.label:
            existing.label = candidate.label
        if candidate.ancestor_path and not existing.ancestor_path:
            existing.ancestor_path = candidate.ancestor_path
        if candidate.branch_family and not existing.branch_family:
            existing.branch_family = candidate.branch_family
        existing.depth_from_root = max(
            int(existing.depth_from_root or 0),
            int(candidate.depth_from_root or 0),
        )

        if int(candidate.parent_distance or 0) > 0:
            if int(existing.parent_distance or 0) <= 0:
                existing.parent_distance = int(candidate.parent_distance or 0)
            else:
                existing.parent_distance = min(
                    int(existing.parent_distance or 0),
                    int(candidate.parent_distance or 0),
                )

        for value in candidate.matched_keywords or []:
            if value not in existing.matched_keywords:
                existing.matched_keywords.append(value)
        for value in candidate.register or []:
            if value not in existing.register:
                existing.register.append(value)

        seen_evidence = {
            (ev.seed, ev.seed_type, ev.match_type, tuple(ev.path))
            for ev in existing.evidence or []
        }
        for ev in candidate.evidence or []:
            key = (ev.seed, ev.seed_type, ev.match_type, tuple(ev.path))
            if key in seen_evidence:
                continue
            existing.evidence.append(ev)
            seen_evidence.add(key)

    def _ancestor_rows(
        self,
        conn: sqlite3.Connection,
        notation: str,
        max_hops: int,
    ) -> List[sqlite3.Row]:
        return conn.execute(
            """
            SELECT ancestor_notation AS notation, distance
            FROM rvk_closure
            WHERE descendant_notation = ?
              AND distance BETWEEN 1 AND ?
            ORDER BY distance ASC
            """,
            [notation, int(max_hops)],
        ).fetchall()

    def _descendant_rows(
        self,
        conn: sqlite3.Connection,
        notation: str,
        max_hops: int,
    ) -> List[sqlite3.Row]:
        return conn.execute(
            """
            SELECT descendant_notation AS notation, distance
            FROM rvk_closure
            WHERE ancestor_notation = ?
              AND distance BETWEEN 1 AND ?
            ORDER BY distance ASC
            """,
            [notation, int(max_hops)],
        ).fetchall()

    def _sibling_rows(
        self,
        conn: sqlite3.Connection,
        notation: str,
        limit: int,
    ) -> List[sqlite3.Row]:
        return conn.execute(
            """
            SELECT DISTINCT sibling.child_notation AS notation, edge.parent_notation AS parent_notation
            FROM rvk_edge edge
            JOIN rvk_edge sibling
              ON sibling.parent_notation = edge.parent_notation
            WHERE edge.child_notation = ?
              AND sibling.child_notation != ?
            ORDER BY sibling.child_notation ASC
            LIMIT ?
            """,
            [notation, notation, int(limit)],
        ).fetchall()

    def _build_query_seeds(
        self,
        keyword_entries: List[Dict[str, str]],
        rvk_anchor_keywords: Optional[List[str]],
        selected_dk_codes: Optional[List[str]],
        llm_analysis: Optional["LlmKeywordAnalysis"],
    ) -> List[GraphSeed]:
        seeds: List[GraphSeed] = []
        seen: Set[Tuple[str, str, str]] = set()

        def add_seed(
            value: str,
            seed_type: str,
            weight: float = 1.0,
            gnd_id: str = "",
            source_keyword: str = "",
        ) -> None:
            clean = self.normalize_seed_text(value)
            if not clean:
                return
            key = (clean.casefold(), str(seed_type or ""), str(gnd_id or ""))
            if key in seen:
                return
            seen.add(key)
            seeds.append(
                GraphSeed(
                    value=clean,
                    seed_type=seed_type,
                    weight=float(weight),
                    gnd_id=str(gnd_id or ""),
                    source_keyword=str(source_keyword or ""),
                )
            )

        for entry in keyword_entries or []:
            keyword = self.normalize_seed_text(entry.get("keyword", ""))
            gnd_id = self.normalize_gnd_id(entry.get("gnd_id", ""))
            if keyword and gnd_id:
                add_seed(keyword, "gnd", weight=1.0, gnd_id=gnd_id, source_keyword=keyword)

        for keyword in rvk_anchor_keywords or []:
            clean_keyword = self.normalize_seed_text(keyword)
            if clean_keyword:
                add_seed(clean_keyword, "anchor", weight=0.9)

        analysis_text = ""
        missing_concepts: List[str] = []
        keyword_chains: List[Dict[str, Any]] = []

        if llm_analysis:
            analysis_text = str(getattr(llm_analysis, "analyse_text", "") or "")
            response_text = str(getattr(llm_analysis, "response_full_text", "") or "")
            if not analysis_text and response_text:
                analysis_text = extract_analyse_text_from_response(response_text) or ""

            missing_concepts = list(getattr(llm_analysis, "missing_concepts", []) or [])
            if not missing_concepts and response_text:
                missing_concepts = extract_missing_concepts_from_response(response_text)

            if response_text:
                keyword_chains = extract_keyword_chains_from_response(response_text)

        if analysis_text:
            for phrase in self.split_reason_into_seed_phrases(analysis_text, max_phrases=4):
                add_seed(phrase, "lexical", weight=0.5)

        for concept in missing_concepts:
            add_seed(concept, "lexical", weight=0.75)

        for chain in keyword_chains:
            for term in chain.get("chain", []) or []:
                add_seed(term, "lexical", weight=0.72)
            reason = str(chain.get("reason", "") or "")
            for phrase in self.split_reason_into_seed_phrases(reason, max_phrases=3):
                add_seed(phrase, "lexical", weight=0.55)

        for hint in self.infer_branch_hints_from_dk_codes(selected_dk_codes or []):
            add_seed(hint, "dk_branch", weight=0.45)

        seeds = self.boost_seed_agreement(seeds)
        seeds.sort(
            key=lambda item: (
                -float(item.weight),
                self.seed_type_rank(item.seed_type),
                item.value.casefold(),
            )
        )
        return seeds[:18]

    def _fetch_direct_seed_hits(
        self,
        seeds: List[GraphSeed],
        limit: int,
    ) -> Dict[str, List[GraphCandidate]]:
        conn = self._connect()
        results: Dict[str, List[GraphCandidate]] = {}

        try:
            for seed in seeds:
                rows: List[sqlite3.Row] = []
                if seed.seed_type == "gnd" and seed.gnd_id:
                    rows = conn.execute(
                        """
                        SELECT
                            n.notation,
                            n.label,
                            n.ancestor_path,
                            n.branch_family,
                            n.depth_from_root,
                            GROUP_CONCAT(DISTINCT cre.register_term) AS register_terms,
                            GROUP_CONCAT(DISTINCT cre.field_tag) AS field_tags,
                            COUNT(*) AS term_hits
                        FROM concept_rvk_edge cre
                        JOIN rvk_node n
                          ON n.notation = cre.notation
                        WHERE cre.gnd_id = ?
                        GROUP BY
                            n.notation,
                            n.label,
                            n.ancestor_path,
                            n.branch_family,
                            n.depth_from_root
                        ORDER BY term_hits DESC, n.depth_from_root DESC, n.notation ASC
                        LIMIT ?
                        """,
                        [seed.gnd_id, int(limit)],
                    ).fetchall()
                elif seed.seed_type in {"anchor", "lexical"}:
                    exact = self._normalize_term_key(seed.value)
                    contains = f"%{exact}%"
                    rows = conn.execute(
                        """
                        SELECT
                            n.notation,
                            n.label,
                            n.ancestor_path,
                            n.branch_family,
                            n.depth_from_root,
                            GROUP_CONCAT(DISTINCT tn.display_term) AS register_terms,
                            GROUP_CONCAT(DISTINCT tr.source_field) AS field_tags,
                            COUNT(DISTINCT tr.term_norm) AS term_hits,
                            MAX(
                                CASE
                                    WHEN tr.term_norm = ? THEN 4
                                    WHEN lower(n.label) = ? THEN 3
                                    WHEN tr.term_norm LIKE ? THEN 2
                                    WHEN lower(n.label) LIKE ? THEN 1
                                    ELSE 0
                                END
                            ) AS lexical_rank
                        FROM rvk_node n
                        LEFT JOIN term_rvk_edge tr
                          ON tr.notation = n.notation
                        LEFT JOIN term_node tn
                          ON tn.term_norm = tr.term_norm
                        WHERE
                            tr.term_norm = ?
                            OR lower(n.label) = ?
                            OR tr.term_norm LIKE ?
                            OR lower(n.label) LIKE ?
                        GROUP BY
                            n.notation,
                            n.label,
                            n.ancestor_path,
                            n.branch_family,
                            n.depth_from_root
                        HAVING lexical_rank > 0
                        ORDER BY lexical_rank DESC, term_hits DESC, n.depth_from_root DESC, n.notation ASC
                        LIMIT ?
                        """,
                        [exact, exact, contains, contains, exact, exact, contains, contains, int(limit)],
                    ).fetchall()
                elif seed.seed_type == "dk_branch":
                    branch_family = seed.value.split(":", 1)[-1].strip().upper()[:1]
                    if branch_family:
                        rows = conn.execute(
                            """
                            SELECT
                                n.notation,
                                n.label,
                                n.ancestor_path,
                                n.branch_family,
                                n.depth_from_root,
                                GROUP_CONCAT(DISTINCT tn.display_term) AS register_terms,
                                '' AS field_tags,
                                COUNT(DISTINCT tr.term_norm) AS term_hits
                            FROM rvk_node n
                            LEFT JOIN term_rvk_edge tr
                              ON tr.notation = n.notation
                            LEFT JOIN term_node tn
                              ON tn.term_norm = tr.term_norm
                            WHERE n.branch_family = ?
                            GROUP BY
                                n.notation,
                                n.label,
                                n.ancestor_path,
                                n.branch_family,
                                n.depth_from_root
                            ORDER BY n.depth_from_root DESC, term_hits DESC, n.notation ASC
                            LIMIT ?
                            """,
                            [branch_family, max(1, int(limit // 2))],
                        ).fetchall()

                if not rows:
                    continue

                seed_hits: List[GraphCandidate] = []
                for row in rows:
                    notation = self.normalize_text(row["notation"])
                    if not notation:
                        continue
                    register_terms = self._parse_csv_values(row["register_terms"])
                    field_tags = self._parse_csv_values(row["field_tags"])
                    exact_register_match = any(term.casefold() == seed.value.casefold() for term in register_terms)
                    contains_register_match = any(seed.value.casefold() in term.casefold() for term in register_terms)
                    lexical_rank = int(row["lexical_rank"] or 0) if "lexical_rank" in row.keys() else 0
                    evidence_weight = float(seed.weight)
                    match_type = "direct_concept" if seed.seed_type == "gnd" else "term"
                    if seed.seed_type == "gnd":
                        evidence_weight += 0.35 + min(int(row["term_hits"] or 0), 3) * 0.1
                        if exact_register_match:
                            evidence_weight += 0.2
                        elif contains_register_match:
                            evidence_weight += 0.1
                    elif seed.seed_type == "dk_branch":
                        evidence_weight += 0.08
                        match_type = "branch"
                    else:
                        evidence_weight += lexical_rank * 0.18
                        if exact_register_match:
                            evidence_weight += 0.12

                    seed_hits.append(
                        GraphCandidate(
                            notation=notation,
                            label=self.normalize_text(row["label"]),
                            branch_family=self.normalize_text(row["branch_family"]),
                            ancestor_path=self.normalize_text(row["ancestor_path"]),
                            depth_from_root=int(row["depth_from_root"] or 0),
                            matched_keywords=[
                                self.normalize_seed_text(seed.source_keyword or seed.value)
                            ],
                            register=register_terms,
                            evidence=[
                                GraphEvidence(
                                    seed=seed.value,
                                    seed_type=seed.seed_type,
                                    match_type=match_type,
                                    weight=round(
                                        evidence_weight + (max(len(field_tags), 1) * 0.04),
                                        4,
                                    ),
                                    path=[seed.value, notation],
                                )
                            ],
                        )
                    )
                if seed_hits:
                    results[seed.value] = seed_hits
        finally:
            conn.close()

        return results

    def _expand_neighborhood(
        self,
        seeds: List[GraphSeed],
        direct_hits: Dict[str, List[GraphCandidate]],
        max_ancestor_hops: int,
        max_descendant_hops: int,
        max_sibling_expansions: int,
        min_context_gain: float,
        min_branch_fit: float,
    ) -> List[GraphCandidate]:
        aggregated: Dict[str, GraphCandidate] = {}
        for seed_hits in direct_hits.values():
            for candidate in seed_hits:
                self._merge_candidate(aggregated, candidate)

        if not aggregated:
            return []

        conn = self._connect()
        try:
            seed_branch_hints = [seed.value for seed in seeds if seed.seed_type == "dk_branch"]
            base_notations = list(aggregated.keys())
            expansion_specs: Dict[str, List[Tuple[str, str, int, str]]] = {}

            for notation in base_notations:
                base_candidate = aggregated[notation]

                for row in self._ancestor_rows(conn, notation, max_ancestor_hops):
                    expansion_specs.setdefault(self.normalize_text(row["notation"]), []).append(
                        ("ancestor", notation, int(row["distance"] or 0), "")
                    )

                descendant_rows = self._descendant_rows(conn, notation, max_descendant_hops)
                for row in descendant_rows[: max(4, max_sibling_expansions * 2)]:
                    expansion_specs.setdefault(self.normalize_text(row["notation"]), []).append(
                        ("child", notation, int(row["distance"] or 0), "")
                    )

                sibling_rows = self._sibling_rows(conn, notation, max_sibling_expansions)
                for row in sibling_rows:
                    expansion_specs.setdefault(self.normalize_text(row["notation"]), []).append(
                        (
                            "sibling",
                            notation,
                            1,
                            self.normalize_text(row["parent_notation"]),
                        )
                    )

            expansion_specs = {
                notation: specs
                for notation, specs in expansion_specs.items()
                if notation
            }
            if not expansion_specs:
                return list(aggregated.values())

            metadata = self._query_node_metadata(conn, list(expansion_specs.keys()))

            for notation, specs in expansion_specs.items():
                node = metadata.get(notation)
                if not node:
                    continue

                local_evidence, local_keywords = self._build_local_node_evidence(node, seeds)
                related_branch_families = {
                    self.normalize_text(aggregated[base_notation].branch_family)
                    for _, base_notation, _, _ in specs
                    if aggregated.get(base_notation)
                }
                branch_fit = self.branch_alignment_score(node.get("branch_family", ""), seed_branch_hints)
                if not branch_fit and node.get("branch_family") in related_branch_families:
                    branch_fit = 1.0

                candidate_evidence: List[GraphEvidence] = list(local_evidence)
                matched_keywords = list(local_keywords)
                parent_distance = 0

                for relation, base_notation, distance, via_parent in specs:
                    base_candidate = aggregated.get(base_notation)
                    if not base_candidate or distance <= 0:
                        continue

                    if relation == "ancestor":
                        decay = 0.62 / float(distance)
                    elif relation == "child":
                        decay = 0.56 / float(distance)
                    else:
                        decay = 0.42 / float(distance)

                    for base_ev in base_candidate.evidence or []:
                        inherited_weight = round(float(base_ev.weight) * decay, 4)
                        if inherited_weight < 0.08:
                            continue
                        path = list(base_ev.path or [])
                        if relation == "sibling" and via_parent:
                            path = path + [via_parent, notation]
                        else:
                            path = path + [notation]
                        candidate_evidence.append(
                            GraphEvidence(
                                seed=base_ev.seed,
                                seed_type=base_ev.seed_type,
                                match_type=relation,
                                weight=inherited_weight,
                                path=path,
                            )
                        )
                    for keyword in base_candidate.matched_keywords or []:
                        if keyword not in matched_keywords:
                            matched_keywords.append(keyword)
                    if relation in {"ancestor", "child"}:
                        if parent_distance <= 0:
                            parent_distance = int(distance)
                        else:
                            parent_distance = min(parent_distance, int(distance))

                support_by_seed = self._extract_candidate_seed_support(
                    GraphCandidate(
                        notation=notation,
                        label=node.get("label", ""),
                        branch_family=node.get("branch_family", ""),
                        ancestor_path=node.get("ancestor_path", ""),
                        depth_from_root=int(node.get("depth_from_root") or 0),
                        matched_keywords=matched_keywords,
                        register=list(node.get("register_terms", []) or []),
                        evidence=candidate_evidence,
                    )
                )
                context_gain = float(sum(support_by_seed.values()))
                if context_gain < float(min_context_gain):
                    continue
                if branch_fit < float(min_branch_fit) and not local_evidence:
                    continue

                if any(spec[0] == "ancestor" for spec in specs):
                    if int(node.get("depth_from_root") or 0) <= 1 and len(support_by_seed) < 2:
                        continue

                expanded_candidate = self._make_candidate_from_node(
                    node=node,
                    evidence=candidate_evidence,
                    matched_keywords=matched_keywords,
                    parent_distance=parent_distance,
                )
                self._merge_candidate(aggregated, expanded_candidate)
        finally:
            conn.close()

        return list(aggregated.values())

    def _score_candidates(
        self,
        seeds: List[GraphSeed],
        candidates: List[GraphCandidate],
        original_abstract: str,
        selected_dk_codes: List[str],
    ) -> List[GraphCandidate]:
        abstract_text = self.normalize_text(original_abstract)
        dk_branch_hints = self.infer_branch_hints_from_dk_codes(selected_dk_codes or [])

        for candidate in candidates:
            evidence = list(candidate.evidence or [])
            seed_support: Dict[str, float] = {}
            for ev in evidence:
                current = seed_support.get(ev.seed, 0.0)
                seed_support[ev.seed] = max(current, float(ev.weight))

            candidate.joint_seed_count = len(seed_support)

            direct_gnd_support = sum(
                float(ev.weight)
                for ev in evidence
                if ev.seed_type == "gnd" and ev.match_type == "direct_concept"
            )
            anchor_support = sum(
                float(ev.weight)
                for ev in evidence
                if ev.seed_type == "anchor"
            )
            lexical_support = sum(
                float(ev.weight)
                for ev in evidence
                if ev.seed_type == "lexical"
            )

            branch_fit = 0.0
            if candidate.branch_family:
                branch_fit += self.branch_alignment_score(candidate.branch_family, dk_branch_hints)
                branch_fit += self.branch_alignment_score(
                    candidate.branch_family,
                    self.infer_branch_hints_from_candidate_text(
                        candidate.label,
                        candidate.ancestor_path,
                        abstract_text,
                    ),
                )

            specificity = self.specificity_score(
                depth=int(candidate.depth_from_root or 0),
                branch_family=candidate.branch_family,
            )

            overgeneral_penalty = 0.0
            if int(candidate.depth_from_root or 0) <= 2:
                overgeneral_penalty += 1.2
            elif int(candidate.depth_from_root or 0) <= 4:
                overgeneral_penalty += 0.6

            if self.has_more_specific_child_with_similar_support(candidate, candidates):
                overgeneral_penalty += 0.9

            branch_mismatch_penalty = 1.0 if branch_fit < 0.20 else 0.0

            if candidate.joint_seed_count >= 3:
                coverage_bonus = 1.8
            elif candidate.joint_seed_count == 2:
                coverage_bonus = 0.9
            else:
                coverage_bonus = 0.0

            evidence_types = {(ev.seed_type, ev.match_type) for ev in evidence}
            explainability_bonus = min(len(evidence_types), 4) * 0.12

            candidate.score = (
                direct_gnd_support * 3.0
                + anchor_support * 2.2
                + lexical_support * 1.1
                + branch_fit * 2.0
                + specificity * 1.4
                + coverage_bonus
                + explainability_bonus
                - overgeneral_penalty
                - branch_mismatch_penalty
            )

        candidates.sort(
            key=lambda item: (
                -float(item.score),
                -int(item.joint_seed_count),
                -int(item.depth_from_root or 0),
                str(item.notation),
            )
        )
        return candidates

    def _select_diverse_shortlist(
        self,
        candidates: List[GraphCandidate],
        limit: int,
    ) -> List[GraphCandidate]:
        selected: List[GraphCandidate] = []
        covered_seeds: Set[str] = set()

        remaining = list(candidates)
        while remaining and len(selected) < limit:
            best_index = None
            best_value = None
            for idx, candidate in enumerate(remaining):
                candidate_seeds = {ev.seed for ev in candidate.evidence or [] if ev.seed}
                new_coverage = len(candidate_seeds - covered_seeds)
                overlap = len(candidate_seeds.intersection(covered_seeds))
                dynamic_value = (
                    float(candidate.score) + new_coverage * 1.4 - overlap * 0.35,
                    int(candidate.joint_seed_count),
                    int(candidate.depth_from_root or 0),
                    -idx,
                )
                if best_value is None or dynamic_value > best_value:
                    best_value = dynamic_value
                    best_index = idx
            if best_index is None:
                break
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            covered_seeds.update(ev.seed for ev in chosen.evidence or [] if ev.seed)

        return selected

    def _format_keyword_centric_results(
        self,
        candidates: List[GraphCandidate],
        keyword_entries: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for entry in keyword_entries or []:
            keyword = self.normalize_seed_text(entry.get("keyword", ""))
            if not keyword:
                continue
            matched = []
            for candidate in candidates:
                if keyword.casefold() in {self.normalize_seed_text(item).casefold() for item in candidate.matched_keywords}:
                    matched.append(candidate.to_pipeline_dict())
            if matched:
                results.append(
                    {
                        "keyword": keyword,
                        "source": "rvk_graph",
                        "search_time_ms": 0.0,
                        "classifications": matched,
                    }
                )
        return results

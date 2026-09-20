"""
stage3_retriever.py
===================

Stage 3 of the Streaming Live RAG pipeline: the *retriever & context-fusion engine*.

Takes the ``sub_queries`` produced by Stage 2::

    [{"query": ..., "intent_type": ..., "topic_label": ..., "topic_id": ...}, ...]

runs a Hybrid Search (Cosine Vector Search via ChromaDB + Keyword Search via Rank-BM25)
fused via Reciprocal Rank Fusion (RRF), and returns only the facts that are **new to
their topic**, while keeping a running per-topic context memory.

Embeddings & Keywords
---------------------
* Vector Search: Uses ChromaDB's built-in ONNX ``all-MiniLM-L6-v2`` (``ONNXMiniLM_L6_V2``).
* Keyword Search: Uses ``rank_bm25.BM25Okapi`` initialized from the ChromaDB corpus on startup.
* Fusion: Combines vector and BM25 candidate ranks using Reciprocal Rank Fusion (RRF).

What "fusion" means here
------------------------
* All sub-queries in one call are embedded and keyword-matched in a single pass.
* Documents already fetched for a topic (in this session) are dropped - dedup is
  **per topic_id**, so the same fact may legitimately appear under two different topics.
* If two sub-queries of the same topic hit the same new document in one call, the
  document is returned once with every query that matched it (``matched_queries``).
* Every call returns the updated ``topic_context`` (all facts held per topic so far).

Session model
-------------
One ``Stage3Retriever`` == one client session's memory. Call ``fork()`` for each
WebSocket connection: forks share the Chroma collection and BM25 index, but maintain
independent memory per session.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from rank_bm25 import BM25Okapi

logger = logging.getLogger("stage3")

DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "streaming_rag_docs"

_DELTA_RE = re.compile(r"^delta_on_topic_(\d+)$")


class Stage3Error(RuntimeError):
    """Raised when the vector store can't be opened or queried."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedDoc:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float  # cosine distance: 0 = identical, ~1 = unrelated
    topic_id: int
    topic_label: str
    intent_type: str
    matched_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "distance": round(self.distance, 4),
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
            "intent_type": self.intent_type,
            "matched_queries": list(self.matched_queries),
        }


@dataclass
class QueryStats:
    query: str
    topic_id: int
    intent_type: str
    retrieved: int = 0  # raw hits returned by search algorithms
    below_relevance: int = 0  # dropped: distance above max_distance
    duplicates_skipped: int = 0  # dropped: already fetched for this topic
    new: int = 0  # kept as new facts

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Stage3Result:
    new_docs: list[RetrievedDoc]
    topic_context: dict[int, dict[str, Any]]
    queries: list[QueryStats]
    latency_ms: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_docs": [d.to_dict() for d in self.new_docs],
            "total_new": len(self.new_docs),
            "topic_context": {str(tid): ctx for tid, ctx in self.topic_context.items()},
            "queries": [q.to_dict() for q in self.queries],
            "latency_ms": self.latency_ms,
            "notes": list(self.notes),
        }


@dataclass
class _Memory:
    """All per-session state, grouped so reset() can replace it in one atomic assignment."""

    fetched_doc_ids: dict[int, set[str]] = field(default_factory=dict)
    docs: dict[int, dict[str, RetrievedDoc]] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Stage3Retriever:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        *,
        top_k: int = 3,
        max_distance: float = 0.8,
        embedding_function: Any = None,
        collection: Any = None,
        bm25: Any = None,
        corpus_ids: list[str] | None = None,
        doc_lookup: dict[str, tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        """
        top_k         - max hits examined per search mode per sub-query.
        max_distance  - drop hits whose vector distance exceeds this (0.8 == similarity < 0.2).
        collection    - (internal, used by fork()) reuse an open Chroma collection.
        bm25          - (internal, used by fork()) reuse a pre-built BM25 model.
        """
        self.top_k = top_k
        self.max_distance = max_distance
        self.collection_name = collection_name
        self._lock = threading.Lock()
        self._memory = _Memory()

        if collection is not None:  # fork(): share heavy resources across sessions
            self._ef = embedding_function
            self._collection = collection
            self._bm25 = bm25
            self._corpus_ids = corpus_ids or []
            self._doc_lookup = doc_lookup or {}
        else:
            self._ef = embedding_function or ONNXMiniLM_L6_V2()
            self._collection = self._open_collection(db_path, collection_name)
            self._warm_up()
            self._init_bm25_corpus()

        self._space = self._detect_space(self._collection)
        self._doc_count = self._collection.count()
        if self._doc_count == 0:
            raise Stage3Error(
                f"Collection '{collection_name}' is empty. Run `python seed_db.py` first."
            )
        logger.info(
            "Stage3 ready: collection=%s docs=%d space=%s top_k=%d max_distance=%.2f hybrid=BM25+Vector",
            collection_name, self._doc_count, self._space, top_k, max_distance,
        )

    # -- setup & initialization --------------------------------------------- #
    def _open_collection(self, db_path: str, name: str):
        try:
            client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
            return client.get_collection(name=name, embedding_function=self._ef)
        except Exception as exc:
            raise Stage3Error(
                f"Could not open collection '{name}' at '{db_path}': {exc}. "
                "Run `python seed_db.py --reset` (from the same working directory as the server)."
            ) from exc

    def _warm_up(self) -> None:
        """Load / download the ONNX model at startup to surface failures early."""
        try:
            self._embed(["warm up"])
        except Exception as exc:
            raise Stage3Error(
                f"Embedding model failed to load: {exc}. Check internet connection/proxy."
            ) from exc

    def _init_bm25_corpus(self) -> None:
        """Fetch all documents from ChromaDB and build the BM25 keyword index."""
        try:
            data = self._collection.get(include=["documents", "metadatas"])
            self._corpus_ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])

            self._doc_lookup = {
                doc_id: (text, dict(meta or {}))
                for doc_id, text, meta in zip(self._corpus_ids, docs, metas)
            }

            tokenized_corpus = [self._tokenize(text) for text in docs]
            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
            else:
                self._bm25 = None
        except Exception as exc:
            logger.warning("Failed to initialize BM25 keyword search: %s", exc)
            self._bm25 = None
            self._corpus_ids = []
            self._doc_lookup = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenizer for BM25 keyword matching."""
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _detect_space(collection: Any) -> str:
        try:
            cfg = getattr(collection, "configuration", None) or {}
            space = (cfg.get("hnsw") or {}).get("space")
            if space:
                return str(space)
        except Exception:
            pass
        return str((getattr(collection, "metadata", None) or {}).get("hnsw:space", "l2"))

    def fork(self) -> "Stage3Retriever":
        """Spawns a new session memory while sharing heavy Chroma & BM25 indexes."""
        return Stage3Retriever(
            collection_name=self.collection_name,
            top_k=self.top_k,
            max_distance=self.max_distance,
            embedding_function=self._ef,
            collection=self._collection,
            bm25=self._bm25,
            corpus_ids=self._corpus_ids,
            doc_lookup=self._doc_lookup,
        )

    # -- state management --------------------------------------------------- #
    @property
    def _fetched_doc_ids(self) -> dict[int, set[str]]:
        return self._memory.fetched_doc_ids

    def reset(self) -> None:
        """Forget all session memory (call on disconnect / new session)."""
        self._memory = _Memory()

    # -- public retrieval API ----------------------------------------------- #
    def retrieve(self, sub_queries: list[dict[str, Any]]) -> Stage3Result:
        started = time.perf_counter()
        notes: list[str] = []
        valid = self._normalise(sub_queries, notes)

        with self._lock:
            mem = self._memory

            if not valid:
                return Stage3Result([], self._snapshot(mem), [], self._ms(started), notes)

            try:
                embeddings = self._embed([sq["query"] for sq in valid])
                raw_vector_results = self._collection.query(
                    query_embeddings=embeddings,
                    n_results=max(1, min(self.top_k, self._doc_count)),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:
                raise Stage3Error(f"Vector search failed: {exc}") from exc

            prior = {sq["topic_id"]: frozenset(mem.fetched_doc_ids.get(sq["topic_id"], ())) for sq in valid}
            batch_new: dict[tuple[int, str], RetrievedDoc] = {}
            stats: list[QueryStats] = []

            for i, sq in enumerate(valid):
                tid = sq["topic_id"]
                st = QueryStats(query=sq["query"], topic_id=tid, intent_type=sq["intent_type"])

                # 1. Vector Search Candidates
                vec_ids = raw_vector_results["ids"][i]
                vec_dists = raw_vector_results["distances"][i]

                vec_rank_map: dict[str, int] = {}
                vec_cos_map: dict[str, float] = {}

                for rank, (doc_id, dist) in enumerate(zip(vec_ids, vec_dists), start=1):
                    cos = self._to_cosine(dist)
                    vec_cos_map[doc_id] = cos
                    if cos <= self.max_distance:
                        vec_rank_map[doc_id] = rank
                    else:
                        st.below_relevance += 1

                # 2. BM25 Keyword Search Candidates
                bm25_rank_map: dict[str, int] = {}
                tokenized_q = self._tokenize(sq["query"])
                if self._bm25 and tokenized_q:
                    scores = self._bm25.get_scores(tokenized_q)
                    top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[: self.top_k]
                    
                    bm_rank = 1
                    for idx in top_indices:
                        if scores[idx] > 0:  # Only include keyword matches with score > 0
                            doc_id = self._corpus_ids[idx]
                            bm25_rank_map[doc_id] = bm_rank
                            bm_rank += 1

                # 3. Reciprocal Rank Fusion (RRF)
                candidate_ids = set(vec_rank_map.keys()) | set(bm25_rank_map.keys())
                st.retrieved = len(candidate_ids)

                rrf_scores: dict[str, float] = {}
                for doc_id in candidate_ids:
                    score = 0.0
                    if doc_id in vec_rank_map:
                        score += 1.0 / (60.0 + vec_rank_map[doc_id])
                    if doc_id in bm25_rank_map:
                        score += 1.0 / (60.0 + bm25_rank_map[doc_id])
                    rrf_scores[doc_id] = score

                # Sort top fused candidate IDs by descending RRF score
                fused_doc_ids = sorted(candidate_ids, key=lambda d_id: rrf_scores[d_id], reverse=True)[: self.top_k]

                # 4. Process and Deduplicate Fused Results
                for doc_id in fused_doc_ids:
                    if doc_id in prior[tid]:
                        st.duplicates_skipped += 1
                        continue

                    key = (tid, doc_id)
                    text, meta = self._doc_lookup.get(doc_id, ("Text unavailable", {}))
                    cos_dist = vec_cos_map.get(doc_id, 0.0)  # Default 0.0 if pure BM25 match

                    if key in batch_new:
                        st.duplicates_skipped += 1
                        existing = batch_new[key]
                        if sq["query"] not in existing.matched_queries:
                            existing.matched_queries.append(sq["query"])
                        existing.distance = min(existing.distance, cos_dist)
                        continue

                    batch_new[key] = RetrievedDoc(
                        id=doc_id,
                        text=text,
                        metadata=meta,
                        distance=cos_dist,
                        topic_id=tid,
                        topic_label=sq["topic_label"],
                        intent_type=sq["intent_type"],
                        matched_queries=[sq["query"]],
                    )
                    st.new += 1

                stats.append(st)

            # Commit to session memory
            new_docs = sorted(batch_new.values(), key=lambda d: (d.topic_id, d.distance))
            for sq in valid:
                mem.labels[sq["topic_id"]] = sq["topic_label"]
            for d in new_docs:
                mem.fetched_doc_ids.setdefault(d.topic_id, set()).add(d.id)
                mem.docs.setdefault(d.topic_id, {})[d.id] = d

            return Stage3Result(new_docs, self._snapshot(mem), stats, self._ms(started), notes)

    # -- internals ---------------------------------------------------------- #
    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() if hasattr(e, "tolist") else [float(x) for x in e] for e in self._ef(texts)]

    def _to_cosine(self, distance: float) -> float:
        if self._space == "l2":
            return distance / 2.0
        return distance

    @staticmethod
    def _ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _normalise(sub_queries: list[dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        for raw in sub_queries or []:
            query = str(raw.get("query", "")).strip()
            intent = str(raw.get("intent_type") or "new_topic")
            topic_id = raw.get("topic_id")
            if topic_id is None and (m := _DELTA_RE.match(intent)):
                topic_id = m.group(1)
            try:
                topic_id = int(topic_id)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                notes.append(f"skipped sub-query without a usable topic_id: {raw!r}")
                continue
            if not query:
                notes.append(f"skipped empty query for topic {topic_id}")
                continue
            valid.append(
                {
                    "query": query,
                    "intent_type": intent,
                    "topic_id": topic_id,
                    "topic_label": str(raw.get("topic_label") or f"topic {topic_id}"),
                }
            )
        return valid

    @staticmethod
    def _snapshot(mem: _Memory) -> dict[int, dict[str, Any]]:
        snapshot: dict[int, dict[str, Any]] = {}
        for tid in sorted(set(mem.labels) | set(mem.docs)):
            docs = mem.docs.get(tid, {})
            snapshot[tid] = {
                "topic_id": tid,
                "label": mem.labels.get(tid, f"topic {tid}"),
                "count": len(docs),
                "doc_ids": list(docs),
                "facts": [{"id": d.id, "text": d.text} for d in docs.values()],
            }
        return snapshot
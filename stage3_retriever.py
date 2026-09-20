"""
stage3_retriever.py
===================

Stage 3 of the Streaming Live RAG pipeline: the *retriever & context-fusion engine*.

Takes the ``sub_queries`` produced by Stage 2::

    [{"query": ..., "intent_type": ..., "topic_label": ..., "topic_id": ...}, ...]

runs a vector search against a local ChromaDB collection, and returns only the facts
that are **new to their topic**, while keeping a running per-topic context memory.

Embeddings
----------
Uses ChromaDB's built-in ONNX ``all-MiniLM-L6-v2`` (``ONNXMiniLM_L6_V2``), which runs on
``onnxruntime`` - **no PyTorch required** (avoids the Windows ``torch`` DLL load error).
The model (~80 MB) is downloaded once to ``~/.cache/chroma/onnx_models`` on first use,
so the first start needs internet access.

What "fusion" means here
------------------------
* All sub-queries in one call are embedded in a single batch.
* Documents already fetched for a topic (in this session) are dropped - dedup is
  **per topic_id**, so the same fact may legitimately appear under two different topics.
* If two sub-queries of the same topic hit the same new document in one call, the
  document is returned once with every query that matched it (``matched_queries``).
* Every call returns the updated ``topic_context`` (all facts held per topic so far).

Session model
-------------
One ``Stage3Retriever`` == one client session's memory. Loading the ONNX model and opening
the DB is the expensive part, so build one instance at server start and call ``fork()``
for each WebSocket connection: forks share the collection but have independent memory.

Thread-safety: ``retrieve()`` is synchronous (Chroma + ONNX are blocking); call it with
``asyncio.to_thread``. Calls are serialised by an internal lock. ``reset()`` swaps the
memory object atomically and never blocks, so it is safe to call from the event loop even
while a retrieval is running in a worker thread.
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
    retrieved: int = 0  # raw hits returned by Chroma
    below_relevance: int = 0  # dropped: distance above max_distance
    duplicates_skipped: int = 0  # dropped: already fetched for this topic (or earlier in batch)
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
            # JSON object keys must be strings
            "topic_context": {str(tid): ctx for tid, ctx in self.topic_context.items()},
            "queries": [q.to_dict() for q in self.queries],
            "latency_ms": self.latency_ms,
            "notes": list(self.notes),
        }


@dataclass
class _Memory:
    """All per-session state, grouped so reset() can replace it in one atomic assignment."""

    fetched_doc_ids: dict[int, set[str]] = field(default_factory=dict)
    docs: dict[int, dict[str, RetrievedDoc]] = field(default_factory=dict)  # insertion-ordered
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
    ) -> None:
        """
        top_k         - max hits examined per sub-query (dedup then removes already-seen ones)
        max_distance  - drop hits whose cosine distance exceeds this (0.8 == similarity < 0.2).
                        Tune with the distances printed by ``python seed_db.py``.
        collection    - (internal, used by fork()) reuse an already-open collection.
        """
        self.top_k = top_k
        self.max_distance = max_distance
        self.collection_name = collection_name
        self._lock = threading.Lock()
        self._memory = _Memory()

        if collection is not None:  # fork(): share the heavy resources
            self._ef = embedding_function
            self._collection = collection
        else:
            self._ef = embedding_function or ONNXMiniLM_L6_V2()
            self._collection = self._open_collection(db_path, collection_name)
            self._warm_up()
        self._space = self._detect_space(self._collection)
        self._doc_count = self._collection.count()
        if self._doc_count == 0:
            raise Stage3Error(
                f"Collection '{collection_name}' is empty. Run `python seed_db.py` first."
            )
        logger.info(
            "Stage3 ready: collection=%s docs=%d space=%s top_k=%d max_distance=%.2f",
            collection_name, self._doc_count, self._space, top_k, max_distance,
        )

    # -- setup -------------------------------------------------------------- #
    def _open_collection(self, db_path: str, name: str):
        try:
            client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
            return client.get_collection(name=name, embedding_function=self._ef)
        except Exception as exc:  # NotFoundError / InvalidCollectionException / config conflicts
            raise Stage3Error(
                f"Could not open collection '{name}' at '{db_path}': {exc}. "
                "Run `python seed_db.py --reset` (from the same working directory as the server)."
            ) from exc

    def _warm_up(self) -> None:
        """Load / download the ONNX model now, so failures surface at startup, not mid-stream."""
        try:
            self._embed(["warm up"])
        except Exception as exc:
            raise Stage3Error(
                f"Embedding model failed to load: {exc}. The ONNX MiniLM model is downloaded on "
                "first use - check your internet connection / proxy and retry."
            ) from exc

    @staticmethod
    def _detect_space(collection: Any) -> str:
        try:  # chromadb >= 1.x
            cfg = getattr(collection, "configuration", None) or {}
            space = (cfg.get("hnsw") or {}).get("space")
            if space:
                return str(space)
        except Exception:
            pass
        return str((getattr(collection, "metadata", None) or {}).get("hnsw:space", "l2"))

    def fork(self) -> "Stage3Retriever":
        """New session memory, same collection + embedding model (cheap)."""
        return Stage3Retriever(
            collection_name=self.collection_name,
            top_k=self.top_k,
            max_distance=self.max_distance,
            embedding_function=self._ef,
            collection=self._collection,
        )

    # -- state -------------------------------------------------------------- #
    @property
    def _fetched_doc_ids(self) -> dict[int, set[str]]:
        """topic_id -> ids of documents already handed downstream in this session."""
        return self._memory.fetched_doc_ids

    def reset(self) -> None:
        """Forget everything (call on disconnect / new session). Never blocks."""
        self._memory = _Memory()

    # -- public API --------------------------------------------------------- #
    def retrieve(self, sub_queries: list[dict[str, Any]]) -> Stage3Result:
        started = time.perf_counter()
        notes: list[str] = []
        valid = self._normalise(sub_queries, notes)

        with self._lock:
            mem = self._memory  # a concurrent reset() swaps self._memory; we finish on the old one

            if not valid:
                return Stage3Result([], self._snapshot(mem), [], self._ms(started), notes)

            try:
                embeddings = self._embed([sq["query"] for sq in valid])
                raw = self._collection.query(
                    query_embeddings=embeddings,
                    n_results=max(1, min(self.top_k, self._doc_count)),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:
                raise Stage3Error(f"Vector search failed: {exc}") from exc

            # What each topic already had *before* this call. Docs first found by an earlier
            # sub-query in this same call are handled by the merge logic below.
            prior = {sq["topic_id"]: frozenset(mem.fetched_doc_ids.get(sq["topic_id"], ())) for sq in valid}
            batch_new: dict[tuple[int, str], RetrievedDoc] = {}
            stats: list[QueryStats] = []

            for i, sq in enumerate(valid):
                tid = sq["topic_id"]
                st = QueryStats(query=sq["query"], topic_id=tid, intent_type=sq["intent_type"])
                ids, docs = raw["ids"][i], raw["documents"][i]
                metas, dists = raw["metadatas"][i], raw["distances"][i]
                for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
                    st.retrieved += 1
                    cos = self._to_cosine(dist)
                    if cos > self.max_distance:
                        st.below_relevance += 1
                        continue
                    key = (tid, doc_id)
                    if doc_id in prior[tid]:
                        st.duplicates_skipped += 1
                        continue
                    if key in batch_new:  # same fact found by two sub-queries of one topic: fuse
                        st.duplicates_skipped += 1
                        existing = batch_new[key]
                        if sq["query"] not in existing.matched_queries:
                            existing.matched_queries.append(sq["query"])
                        existing.distance = min(existing.distance, cos)
                        continue
                    batch_new[key] = RetrievedDoc(
                        id=doc_id,
                        text=text,
                        metadata=dict(meta or {}),
                        distance=cos,
                        topic_id=tid,
                        topic_label=sq["topic_label"],
                        intent_type=sq["intent_type"],
                        matched_queries=[sq["query"]],
                    )
                    st.new += 1
                stats.append(st)

            # Commit to session memory, best matches first within each topic.
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
        """Normalise Chroma's distance to cosine distance (MiniLM vectors are unit length)."""
        if self._space == "l2":  # Chroma reports squared L2: ||a-b||^2 = 2 - 2cos
            return distance / 2.0
        return distance  # "cosine" and "ip" (1 - dot) are already cosine distance for unit vectors

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

"""
stage2_decomposer.py
====================

Stage 2 of the Streaming Live RAG pipeline: the *query decomposer*.

Takes the accumulated transcript buffer (when Stage 1 says PROVISIONAL_RETRIEVE
or the utterance ends) and asks an LLM on Groq to break it into
retrieval-ready sub-queries, each tagged with an ``intent_type``:

    "new_topic"            - a topic not covered by earlier sub-queries
    "delta_on_topic_<N>"   - refines / extends already-known topic <N>

Key behaviours
--------------
* Structured output is defined by Pydantic models and enforced three ways:
    1. ``response_format=json_schema`` (strict) when the provider supports it,
    2. fallback ``json_object`` mode, then plain-prompt JSON,
    3. *always* validated with ``model_validate_json(strict=True)``; on failure
       the error is fed back to the model for a bounded number of retries.
* The decomposer is **stateful per session**: it keeps a topic registry and the
  set of already-issued queries, so repeated provisional calls on a growing
  transcript yield only *new* sub-queries (deduplicated), tagged as deltas
  where appropriate.

Environment
-----------
    GROQ_API_KEY         required
    GROQ_MODEL           optional, default "llama-3.3-70b-versatile"
    GROQ_REFERER         optional, sent as the HTTP-Referer header
    GROQ_APP_TITLE       optional, sent as the X-Title header
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("stage2")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

_INTENT_RE = re.compile(r"^(?:new_topic|delta_on_topic_\d+)$")


# --------------------------------------------------------------------------- #
# Structured-output schema (this is what the LLM must emit)
# --------------------------------------------------------------------------- #
class SubQuery(BaseModel):
    """One retrieval-ready sub-query. All fields required (strict-schema friendly)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Self-contained search query; no pronouns or references to 'it'/'that'.")
    intent_type: str = Field(
        description='Either "new_topic" or "delta_on_topic_<N>" where <N> is the id of a known topic.'
    )
    topic_label: str = Field(description="Short (2-5 words) label of the topic this query belongs to.")

    @field_validator("query", "topic_label")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("intent_type")
    @classmethod
    def _valid_intent(cls, v: str) -> str:
        v = v.strip()
        if not _INTENT_RE.match(v):
            raise ValueError('intent_type must be "new_topic" or "delta_on_topic_<N>" (N = integer topic id)')
        return v


class DecompositionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sub_queries: list[SubQuery] = Field(
        description="New sub-queries only. Empty list if the transcript adds nothing new to retrieve."
    )


class ResolvedSubQuery(SubQuery):
    """SubQuery after the decomposer has assigned a concrete topic id."""

    topic_id: int


@dataclass
class DecompositionResult:
    sub_queries: list[ResolvedSubQuery]
    latency_ms: int
    model: str
    mode: str  # "json_schema" | "json_object" | "prompt_only"
    attempts: int = 1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sub_queries": [sq.model_dump() for sq in self.sub_queries],
            "latency_ms": self.latency_ms,
            "model": self.model,
            "mode": self.mode,
            "attempts": self.attempts,
            "notes": self.notes,
        }


class DecomposerError(RuntimeError):
    """Raised when no valid structured output could be obtained."""


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are the query-decomposition stage of a live retrieval-augmented (RAG) system.
You receive a partial, possibly messy speech transcript that is still being spoken. Convert it into
sub-queries for a search index.

Rules:
1. Output ONLY sub-queries that are NEW compared with `already_issued_queries`. Never repeat or
   trivially rephrase an issued query. If nothing new is worth retrieving, return an empty list.
2. Each query must be self-contained: resolve pronouns and references ("it", "that company") using
   the transcript and known topics. Keep queries short and keyword-rich (a search query, not a sentence
   of chat).
3. Split compound questions into separate atomic sub-queries (max 5 in total).
4. Tag every sub-query with `intent_type`:
   - "new_topic" if it concerns a subject not in `known_topics`.
   - "delta_on_topic_<N>" if it refines, narrows, compares, or extends known topic id <N>
     (e.g. topic 1 is "Tesla batteries" and the speaker now asks about their cost -> "delta_on_topic_1").
   Only use ids that appear in `known_topics`. For a brand-new topic use exactly "new_topic".
5. `topic_label`: 2-5 word label of the topic. For a delta, reuse the known topic's label. Sub-queries
   that belong to the same new topic must share the same label.
6. Ignore filler, greetings, and disfluencies. The tail of the transcript may be cut off mid-sentence;
   do not invent content beyond what was said.
7. Respond with a single JSON object and nothing else."""


def _schema_hint() -> str:
    return (
        "Respond with ONLY a JSON object of this exact shape (no markdown, no commentary):\n"
        '{"sub_queries": [{"query": "...", "intent_type": "new_topic | delta_on_topic_<N>", '
        '"topic_label": "..."}]}'
    )


def _normalize(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", q.lower())).strip()


def _extract_json(text: str) -> str:
    """Tolerate ```json fences and leading/trailing chatter from non-strict providers."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


# --------------------------------------------------------------------------- #
# Decomposer
# --------------------------------------------------------------------------- #
class Stage2Decomposer:
    _MODES = ("json_schema", "json_object", "prompt_only")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        client: Optional[AsyncOpenAI] = None,
        max_validation_retries: int = 2,
        max_sub_queries: int = 5,
        timeout: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        if client is None:
            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise DecomposerError("GROQ_API_KEY is not set")
            client = AsyncOpenAI(
                base_url=GROQ_BASE_URL,
                api_key=api_key,
                timeout=timeout,
                max_retries=2,  # SDK-level backoff for 429 rate limits / transient 5xx
                default_headers={
                    "HTTP-Referer": os.getenv("GROQ_REFERER", "http://localhost:8000"),
                    "X-Title": os.getenv("GROQ_APP_TITLE", "Streaming Live RAG"),
                },
            )
        self._client = client
        self._max_val_retries = max_validation_retries
        self._max_sq = max_sub_queries
        self._temperature = temperature
        self._mode_idx = 0  # sticky: once a mode is rejected we don't retry it

        # Session state
        self._topics: dict[int, str] = {}  # topic_id -> label
        self._issued: dict[str, int] = {}  # normalized query -> topic_id
        self._issued_text: list[str] = []  # original query strings, in issue order
        self._next_topic_id = 1

    # -- session state ------------------------------------------------------ #
    @property
    def topics(self) -> dict[int, str]:
        return dict(self._topics)

    def reset(self) -> None:
        self._topics.clear()
        self._issued.clear()
        self._issued_text.clear()
        self._next_topic_id = 1

    # -- public API --------------------------------------------------------- #
    async def decompose(self, transcript: str, *, final: bool = False) -> DecompositionResult:
        transcript = transcript.strip()
        if not transcript:
            return DecompositionResult([], 0, self.model, self._MODES[self._mode_idx])

        started = time.perf_counter()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_payload(transcript, final)},
        ]

        parsed: Optional[DecompositionOutput] = None
        last_err = "unknown error"
        attempts = 0
        for attempt in range(1, self._max_val_retries + 2):
            attempts = attempt
            raw = await self._call_llm(messages)
            try:
                parsed = DecompositionOutput.model_validate_json(_extract_json(raw), strict=True)
                break
            except (ValidationError, ValueError) as exc:
                last_err = str(exc)
                logger.warning("Stage2 invalid output (attempt %d): %s", attempt, last_err[:300])
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": "Your previous reply was invalid: "
                        f"{last_err[:500]}\n{_schema_hint()}",
                    },
                ]
        if parsed is None:
            raise DecomposerError(f"No valid structured output after {attempts} attempts: {last_err[:300]}")

        resolved, notes = self._resolve(parsed)
        return DecompositionResult(
            sub_queries=resolved,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=self.model,
            mode=self._MODES[self._mode_idx],
            attempts=attempts,
            notes=notes,
        )

    # -- prompt construction ------------------------------------------------ #
    def _build_user_payload(self, transcript: str, final: bool) -> str:
        payload = {
            "transcript": transcript,
            "is_final_utterance": final,
            "known_topics": [{"id": i, "label": lbl} for i, lbl in self._topics.items()],
            "already_issued_queries": self._issued_text[-25:],
        }
        return json.dumps(payload, ensure_ascii=False)

    # -- LLM call with graceful capability fallback ------------------------- #
    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        while True:
            mode = self._MODES[self._mode_idx]
            msgs = messages
            if mode != "json_schema":  # no server-side schema: spell the shape out in the prompt
                msgs = [{"role": "system", "content": f"{messages[0]['content']}\n\n{_schema_hint()}"}, *messages[1:]]
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=msgs,
                temperature=self._temperature,
                max_tokens=700,
            )
            if mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "subquery_decomposition",
                        "strict": True,
                        "schema": DecompositionOutput.model_json_schema(),
                    },
                }
            elif mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = await self._client.chat.completions.create(**kwargs)
            except (openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError) as exc:
                if self._mode_idx < len(self._MODES) - 1:
                    logger.warning("Mode %s rejected (%s); falling back.", mode, getattr(exc, "message", exc))
                    self._mode_idx += 1
                    continue
                raise DecomposerError(f"Groq rejected the request: {exc}") from exc
            except openai.APIError as exc:
                raise DecomposerError(f"Groq API error: {exc}") from exc

            content = (resp.choices[0].message.content or "") if resp.choices else ""
            if not content.strip():
                # Some providers silently ignore json_schema and return empty content.
                if self._mode_idx < len(self._MODES) - 1:
                    logger.warning("Mode %s returned empty content; falling back.", mode)
                    self._mode_idx += 1
                    continue
                raise DecomposerError("Model returned empty content")
            return content

    # -- post-processing: topic ids, dedupe, delta validation ---------------- #
    def _resolve(self, out: DecompositionOutput) -> tuple[list[ResolvedSubQuery], list[str]]:
        resolved: list[ResolvedSubQuery] = []
        notes: list[str] = []
        batch_labels: dict[str, int] = {}  # label -> topic id created in *this* batch

        for sq in out.sub_queries[: self._max_sq]:
            key = _normalize(sq.query)
            if not key or key in self._issued:
                notes.append(f"dropped duplicate: {sq.query!r}")
                continue

            intent = sq.intent_type
            topic_id: int
            if intent == "new_topic":
                lbl_key = sq.topic_label.lower()
                if lbl_key in batch_labels:
                    topic_id = batch_labels[lbl_key]
                else:
                    topic_id = self._next_topic_id
                    self._next_topic_id += 1
                    self._topics[topic_id] = sq.topic_label
                    batch_labels[lbl_key] = topic_id
            else:
                topic_id = int(intent.rsplit("_", 1)[1])
                if topic_id not in self._topics:
                    notes.append(f"unknown topic {topic_id} in {intent!r}; treated as new_topic")
                    intent = "new_topic"
                    topic_id = self._next_topic_id
                    self._next_topic_id += 1
                    self._topics[topic_id] = sq.topic_label
                    batch_labels[sq.topic_label.lower()] = topic_id

            self._issued[key] = topic_id
            self._issued_text.append(sq.query)
            resolved.append(
                ResolvedSubQuery(
                    query=sq.query, intent_type=intent, topic_label=self._topics[topic_id], topic_id=topic_id
                )
            )
        return resolved, notes

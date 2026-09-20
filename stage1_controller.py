"""
stage1_controller.py
====================

Stage 1 of the Streaming Live RAG pipeline: the *gatekeeper*.

Consumes text chunks from a live transcript stream, debounces them (500 ms by
default) and, each time the stream goes quiet, classifies the accumulated
buffer into one of three decisions:

    WAIT                  - clause is syntactically incomplete, or nothing new/stable yet
    SUPPRESS              - filler / small talk, not worth retrieving for
    PROVISIONAL_RETRIEVE  - a stable cluster of *new* entities was detected

Design notes
------------
* Debouncing: every chunk restarts a 500 ms timer. When the timer expires
  without another chunk, the buffer is evaluated ("debounce" trigger). A
  ``max_wait`` cap (lodash-style) guarantees an evaluation at least every
  ``max_wait_ms`` even if the speaker never pauses ("max_wait" trigger).
* Entity stability: an entity candidate is *stable* when at least one holds:
    - it is not at the trailing edge of the buffer (more words followed it), or
    - the stream has been quiet for the full debounce window (it survived
      500 ms without being extended), or
    - the buffer ends in terminal punctuation, or
    - it was already seen unchanged in the previous evaluation.
  Trailing entities are the risky ones ("New" -> "New York" -> "New York Times").
* Syntactic completeness: a buffer that ends in a dangling function word
  ("... about", "... and", "what is the") or a trailing comma/ellipsis is
  considered an incomplete clause -> WAIT.
* NLU backend: spaCy (``en_core_web_sm``) when available, otherwise a
  rule-based fallback with the same interface, so the pipeline never hard-fails
  on a missing model.
* End-of-utterance: ``end_utterance()`` flushes immediately and emits a
  ``final=True`` event. Any non-filler final buffer yields
  PROVISIONAL_RETRIEVE so Stage 2 always sees the complete utterance.

Only the standard library is required; spaCy is optional.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Iterable, Optional, Protocol

logger = logging.getLogger("stage1")


# --------------------------------------------------------------------------- #
# Public data types
# --------------------------------------------------------------------------- #
class Decision(str, Enum):
    WAIT = "WAIT"
    SUPPRESS = "SUPPRESS"
    PROVISIONAL_RETRIEVE = "PROVISIONAL_RETRIEVE"


@dataclass(frozen=True)
class Stage1Event:
    """Everything downstream stages / loggers need to know about one decision."""

    decision: Decision
    reason: str
    buffer: str
    trigger: str  # "debounce" | "max_wait" | "end_of_utterance"
    utterance_id: int
    final: bool = False
    stable_entities: tuple[str, ...] = ()
    new_entities: tuple[str, ...] = ()
    ts: float = field(default_factory=time.time)

    @property
    def should_decompose(self) -> bool:
        """Stage 2 runs on PROVISIONAL_RETRIEVE (which also covers non-filler EOU)."""
        return self.decision is Decision.PROVISIONAL_RETRIEVE

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "buffer": self.buffer,
            "trigger": self.trigger,
            "utterance_id": self.utterance_id,
            "final": self.final,
            "stable_entities": list(self.stable_entities),
            "new_entities": list(self.new_entities),
            "ts": self.ts,
        }


@dataclass(frozen=True)
class EntityCandidate:
    text: str
    label: str
    at_tail: bool  # candidate touches the last content token of the buffer

    @property
    def key(self) -> str:
        return re.sub(r"\s+", " ", self.text.lower()).strip()


@dataclass
class Analysis:
    entities: list[EntityCandidate]
    is_filler: bool
    is_dangling: bool  # ends mid-clause (function word / comma / ellipsis)
    has_predicate: bool  # contains a verb (informational; used in reasons)
    ends_terminal: bool  # ends with . ? ! (or similar)


# --------------------------------------------------------------------------- #
# Lexical resources
# --------------------------------------------------------------------------- #
_FILLER_PHRASES = [
    "you know", "i mean", "let me think", "let me see", "one second", "one sec",
    "hold on", "sort of", "kind of", "sounds good", "got it", "i see", "makes sense",
    "thank you", "thanks", "hello", "hey", "hi", "good morning", "good afternoon",
    "good evening", "bye", "goodbye", "um", "uh", "uhh", "umm", "hmm", "hm", "mm",
    "mhm", "ah", "oh", "er", "erm", "okay", "ok", "yeah", "yep", "yes", "no", "nope",
    "right", "so", "well", "like", "alright", "anyway", "cool", "great", "nice",
    "sure", "fine", "actually", "basically", "please", "and", "but", "then",
    "sorry", "pardon", "wow", "lol", "haha", "the", "a", "just", "really", "very",
]
_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, _FILLER_PHRASES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_SMALL_TALK_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bhow(?:'s| is| are) (?:it going|you|things|everyone|your day)\b",
        r"\bnice to (?:meet|see) you\b",
        r"\bcan you hear me\b",
        r"\b(?:is|are) (?:this|you) (?:working|there|on)\b",
        r"\btesting(?:,)? (?:testing|one|1)\b",
        r"\bwhat'?s up\b",
        r"\bhave a (?:good|great|nice) (?:day|one|weekend)\b",
        r"\b(?:the )?weather (?:is|has been)\b",
        r"\bsee you (?:later|soon|tomorrow)\b",
        r"\bcan you (?:see|hear) (?:my|the) screen\b",
    )
]

# Last-word cues that the clause is still open.
_DANGLING_TAIL_WORDS = {
    # conjunctions / subordinators
    "and", "or", "but", "nor", "because", "since", "although", "though", "while",
    "whereas", "if", "unless", "that", "which", "who", "whom", "whose", "when",
    "where", "whether", "so", "then", "plus",
    # determiners / possessives
    "the", "a", "an", "this", "these", "those", "my", "your", "our", "their",
    "his", "her", "its", "some", "any", "each", "every", "no",
    # prepositions / particles
    "of", "to", "in", "on", "at", "for", "with", "about", "from", "by", "into",
    "onto", "over", "under", "between", "among", "regarding", "versus", "vs",
    "than", "as", "like", "across", "through", "after", "before", "during",
    # auxiliaries / copulas
    "is", "are", "was", "were", "am", "be", "been", "being", "do", "does", "did",
    "has", "have", "had", "will", "would", "can", "could", "should", "shall",
    "may", "might", "must", "'s",
    # hedges
    "um", "uh", "uhh", "umm", "hmm", "er", "erm",
}
_DANGLING_POS = {"ADP", "CCONJ", "SCONJ", "DET", "AUX", "PART"}

_GENERIC_HEADS = {
    "thing", "things", "stuff", "something", "anything", "everything", "nothing",
    "one", "ones", "way", "ways", "lot", "lots", "bit", "kind", "sort", "time",
    "times", "question", "questions", "idea", "ideas", "part", "parts", "case",
    "example", "sense", "point", "moment", "minute", "second", "day", "today",
    "let", "problem", "issue",
}
_IGNORED_ENT_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "TIME"}

_SENTENCE_OPENERS = {
    "what", "how", "why", "when", "where", "who", "which", "can", "could", "would",
    "should", "do", "does", "did", "is", "are", "was", "were", "tell", "explain",
    "show", "give", "compare", "describe", "summarize", "summarise", "find",
    "list", "define", "walk", "please", "so", "well", "okay", "ok", "and", "but",
    "the", "a", "an", "i", "we", "you", "it", "he", "she", "they", "let", "let's",
    "also", "then", "now", "yeah", "yes", "no", "hey", "hi", "hello", "um", "uh",
}
_STOPWORDS = _SENTENCE_OPENERS | _DANGLING_TAIL_WORDS | {
    "me", "us", "them", "him", "my", "more", "much", "many", "most", "other",
}
_TERMINAL_RE = re.compile(r"[.?!]['\")\]]*\s*$")
_DANGLING_PUNCT_RE = re.compile(r"(?:[,:;\-\u2013\u2014]|\.\.\.|\u2026)\s*$")


# --------------------------------------------------------------------------- #
# NLU backends
# --------------------------------------------------------------------------- #
class Analyzer(Protocol):
    name: str

    def analyze(self, text: str) -> Analysis: ...


def _is_filler_text(text: str) -> bool:
    """True if nothing but filler words / small talk remains after stripping them."""
    remainder, had_small_talk = text, False
    for pattern in _SMALL_TALK_RES:
        remainder, n = pattern.subn(" ", remainder)
        had_small_talk = had_small_talk or n > 0
    remainder = _FILLER_RE.sub(" ", remainder)
    if had_small_talk:
        # Small talk is only filler if no real content words survive around it.
        return len(re.findall(r"[A-Za-z0-9]{3,}", remainder)) <= 1
    return not re.search(r"[A-Za-z0-9]", remainder)


def _dangling(text: str, last_word: str, last_pos: Optional[str]) -> bool:
    if _DANGLING_PUNCT_RE.search(text):
        return True
    if _TERMINAL_RE.search(text):
        return False
    if last_word.lower() in _DANGLING_TAIL_WORDS:
        return True
    return last_pos in _DANGLING_POS if last_pos else False


class RuleBasedAnalyzer:
    """Dependency-free fallback: regex heuristics for entities & completeness."""

    name = "rules"

    _CAP_RUN = re.compile(r"\b(?:[A-Z][A-Za-z0-9\-']*|[A-Z]{2,})(?:\s+(?:of\s+|the\s+|and\s+)?[A-Z][A-Za-z0-9\-']*)*")
    _TECH_TOKEN = re.compile(r"\b(?:[a-z]+[A-Z]\w*|\w*\d+\w*[A-Za-z]\w*|[A-Z]{2,}\d*)\b")
    _AFTER_PREP = re.compile(
        r"\b(?:about|regarding|on|of|for|between|with|versus|vs\.?|compare|explain|define|is|are)\s+"
        r"((?:[A-Za-z][\w\-']*\s+){0,3}[A-Za-z][\w\-']*)",
        re.IGNORECASE,
    )

    def analyze(self, text: str) -> Analysis:
        stripped = text.strip()
        words = re.findall(r"[\w'\-]+", stripped)
        last_word = words[-1] if words else ""
        ends_terminal = bool(_TERMINAL_RE.search(stripped))
        core = re.sub(r"[\s.?!,;:\u2026\-\"')\]]+$", "", stripped)

        spans: dict[str, EntityCandidate] = {}

        def add(span_text: str, end_idx: int, label: str) -> None:
            toks = [t for t in re.findall(r"[\w'\-]+", span_text)]
            while toks and toks[0].lower() in _STOPWORDS:
                toks.pop(0)
            while toks and toks[-1].lower() in _STOPWORDS:
                toks.pop()
            if not toks or all(t.lower() in _STOPWORDS or t.lower() in _GENERIC_HEADS for t in toks):
                return
            clean = " ".join(toks)
            cand = EntityCandidate(clean, label, at_tail=end_idx >= len(core))
            spans.setdefault(cand.key, cand)

        for m in self._CAP_RUN.finditer(stripped):
            add(m.group(0), m.end(), "PROPER")
        for m in self._TECH_TOKEN.finditer(stripped):
            add(m.group(0), m.end(), "TECH")
        for m in self._AFTER_PREP.finditer(stripped):
            add(m.group(1), m.end(1), "TOPIC")

        # Drop candidates that are substrings of longer candidates.
        keys = list(spans)
        for k in keys:
            if any(k != o and k in o for o in keys):
                spans.pop(k, None)

        return Analysis(
            entities=list(spans.values()),
            is_filler=_is_filler_text(stripped),
            is_dangling=_dangling(stripped, last_word, None),
            has_predicate=True,  # unknowable without a tagger; don't block on it
            ends_terminal=ends_terminal,
        )


class SpacyAnalyzer:
    """spaCy-backed analyzer (NER + noun chunks + POS/dependency parse)."""

    name = "spacy"
    _lock = threading.Lock()  # nlp() is not guaranteed thread-safe; serialise calls

    def __init__(self, model: str = "en_core_web_sm") -> None:
        import spacy  # imported lazily so the module works without spaCy

        self._spacy = spacy
        self._nlp = spacy.load(model, disable=["lemmatizer"])

    def analyze(self, text: str) -> Analysis:
        stripped = text.strip()
        with self._lock:
            doc = self._nlp(stripped)

        content_idx = [t.i for t in doc if not (t.is_punct or t.is_space)]
        last_content = content_idx[-1] if content_idx else -1
        last_tok = doc[last_content] if last_content >= 0 else None

        candidates: list = []  # spaCy Spans
        for ent in doc.ents:
            if ent.label_ not in _IGNORED_ENT_LABELS:
                candidates.append(ent)
        for chunk in doc.noun_chunks:
            start = chunk.start
            while start < chunk.end and doc[start].pos_ in {"DET", "PRON", "NUM"}:
                start += 1  # drop "the", "my", "their", "three" - but keep "Tesla's"
            if start >= chunk.end:
                continue
            span = doc[start : chunk.end]
            head = chunk.root
            if head.pos_ not in {"NOUN", "PROPN"} or head.lemma_.lower() in _GENERIC_HEADS:
                continue
            if head.text.lower() in _GENERIC_HEADS or all(t.is_stop for t in span):
                continue
            candidates.append(span)

        entities: list[EntityCandidate] = []
        for span in self._spacy.util.filter_spans(candidates):
            toks = [t for t in span if not t.is_punct]
            while toks and toks[0].is_stop and toks[0].pos_ in {"DET", "ADP", "PRON"}:
                toks.pop(0)
            if not toks:
                continue
            label = span.label_ or ("PROPN" if span.root.pos_ == "PROPN" else "NOUN")
            entities.append(
                EntityCandidate(
                    text=doc[toks[0].i : toks[-1].i + 1].text,
                    label=label,
                    at_tail=(span.end - 1) >= last_content >= 0 and toks[-1].i >= last_content,
                )
            )

        return Analysis(
            entities=entities,
            is_filler=_is_filler_text(stripped)
            or not any(
                t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "NUM"} and not t.is_stop
                for t in doc
            ),
            is_dangling=_dangling(
                stripped, last_tok.text if last_tok else "", last_tok.pos_ if last_tok else None
            ),
            has_predicate=any(t.pos_ in {"VERB", "AUX"} for t in doc),
            ends_terminal=bool(_TERMINAL_RE.search(stripped)),
        )


class NLUClassifier:
    """Loads the best available analyzer (spaCy preferred, rules as fallback)."""

    def __init__(self, spacy_model: str = "en_core_web_sm", force_rules: bool = False) -> None:
        self.analyzer: Analyzer
        if force_rules:
            self.analyzer = RuleBasedAnalyzer()
        else:
            try:
                self.analyzer = SpacyAnalyzer(spacy_model)
            except Exception as exc:  # ImportError, OSError (model missing), ...
                logger.warning("spaCy unavailable (%s); using rule-based NLU fallback.", exc)
                self.analyzer = RuleBasedAnalyzer()
        logger.info("Stage1 NLU backend: %s", self.analyzer.name)

    @property
    def backend(self) -> str:
        return self.analyzer.name

    def analyze(self, text: str) -> Analysis:
        return self.analyzer.analyze(text)


# --------------------------------------------------------------------------- #
# Debouncer
# --------------------------------------------------------------------------- #
class Debouncer:
    """
    Trailing-edge debouncer with a max-wait cap, built on asyncio.

    ``trigger()`` (re)starts the timer. When it expires the callback receives
    ``"debounce"`` (quiet for the full delay) or ``"max_wait"`` (forced because
    the stream never paused long enough).
    """

    def __init__(
        self,
        callback: Callable[[str], Awaitable[None]],
        delay: float = 0.5,
        max_wait: Optional[float] = 2.5,
    ) -> None:
        self._callback = callback
        self._delay = delay
        self._max_wait = max_wait
        self._timer: Optional[asyncio.Task] = None
        self._burst_started: Optional[float] = None
        self._running: set[asyncio.Task] = set()

    def trigger(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._burst_started is None:
            self._burst_started = now
        if self._timer is not None:
            self._timer.cancel()

        wait, kind = self._delay, "debounce"
        if self._max_wait is not None:
            remaining = self._burst_started + self._max_wait - now
            if remaining < self._delay:
                wait, kind = max(0.0, remaining), "max_wait"
        self._timer = loop.create_task(self._sleep_then_fire(wait, kind))

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._burst_started = None

    async def aclose(self) -> None:
        self.cancel()
        for task in list(self._running):
            task.cancel()
        if self._running:
            await asyncio.gather(*self._running, return_exceptions=True)

    async def _sleep_then_fire(self, wait: float, kind: str) -> None:
        await asyncio.sleep(wait)
        # Timer has fired: detach it *before* running the callback so a chunk that
        # arrives during evaluation starts a fresh timer instead of cancelling us.
        self._timer = None
        self._burst_started = None
        task = asyncio.current_task()
        assert task is not None
        self._running.add(task)
        try:
            await self._callback(kind)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Debounce callback failed")
        finally:
            self._running.discard(task)


# --------------------------------------------------------------------------- #
# Stage 1 controller
# --------------------------------------------------------------------------- #
EventCallback = Callable[[Stage1Event], Awaitable[None]]


class Stage1Controller:
    """
    Feed it chunks; it calls ``on_event`` with a :class:`Stage1Event` whenever the
    debouncer fires (or the utterance ends).

    Example
    -------
    >>> ctrl = Stage1Controller(on_event=handler)
    >>> ctrl.feed("Tell me about")
    >>> ctrl.feed("Tesla's battery supply chain")
    >>> await ctrl.end_utterance()
    """

    def __init__(
        self,
        on_event: EventCallback,
        *,
        debounce_ms: int = 500,
        max_wait_ms: Optional[int] = 2500,
        classifier: Optional[NLUClassifier] = None,
        min_cluster_size: int = 1,
        min_new_entities: int = 1,
    ) -> None:
        self._on_event = on_event
        self._clf = classifier or NLUClassifier()
        self._min_cluster = min_cluster_size
        self._min_new = min_new_entities
        self._debouncer = Debouncer(
            self._on_timer,
            delay=debounce_ms / 1000,
            max_wait=None if max_wait_ms is None else max_wait_ms / 1000,
        )
        self._eval_lock = asyncio.Lock()
        self._buffer = ""
        self._utterance_id = 1
        self._prev_seen: set[str] = set()  # entity keys present in previous evaluation
        self._retrieved: set[str] = set()  # entity keys already handed to Stage 2

    # -- public API --------------------------------------------------------- #
    @property
    def buffer(self) -> str:
        return self._buffer

    @property
    def utterance_id(self) -> int:
        return self._utterance_id

    @property
    def backend(self) -> str:
        return self._clf.backend

    def feed(self, chunk: str) -> None:
        """Append a chunk to the utterance buffer and (re)start the debounce timer."""
        if not chunk or not chunk.strip():
            return
        self._buffer = self._join(self._buffer, chunk)
        self._debouncer.trigger()

    async def end_utterance(self) -> None:
        """Flush immediately (no waiting) and start a fresh utterance."""
        self._debouncer.cancel()
        await self._evaluate("end_utterance", final=True)
        self._reset_utterance()

    def reset(self) -> None:
        """Discard the current utterance without evaluating it."""
        self._debouncer.cancel()
        self._reset_utterance()

    async def close(self) -> None:
        await self._debouncer.aclose()

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _join(buffer: str, chunk: str) -> str:
        if not buffer:
            return chunk.strip()
        if buffer[-1].isspace() or chunk[0].isspace() or chunk[0] in ".,;:!?)'\u2019":
            return (buffer + chunk).strip() if chunk[0].isspace() else buffer + chunk
        return buffer + " " + chunk

    def _reset_utterance(self) -> None:
        self._buffer = ""
        self._prev_seen.clear()
        self._retrieved.clear()
        self._utterance_id += 1

    async def _on_timer(self, kind: str) -> None:
        await self._evaluate(kind, final=False)

    async def _evaluate(self, kind: str, final: bool) -> None:
        async with self._eval_lock:
            text = self._buffer.strip()
            if not text:
                return
            analysis = await asyncio.to_thread(self._clf.analyze, text)
            event = self._decide(text, analysis, kind, final)
            try:
                await self._on_event(event)
            except Exception:
                logger.exception("Stage1 on_event handler failed")

    def _decide(self, text: str, a: Analysis, kind: str, final: bool) -> Stage1Event:
        quiet = kind in ("debounce", "end_utterance")
        stable: dict[str, str] = {}
        for cand in a.entities:
            is_stable = (
                not cand.at_tail
                or quiet
                or a.ends_terminal
                or cand.key in self._prev_seen
            )
            if is_stable:
                stable[cand.key] = cand.text
        self._prev_seen = {c.key for c in a.entities}
        new_keys = [k for k in stable if k not in self._retrieved]

        def event(decision: Decision, reason: str, new: Iterable[str] = ()) -> Stage1Event:
            return Stage1Event(
                decision=decision,
                reason=reason,
                buffer=text,
                trigger=kind,
                utterance_id=self._utterance_id,
                final=final,
                stable_entities=tuple(stable.values()),
                new_entities=tuple(stable[k] for k in new),
            )

        # 1) Filler / small talk always wins.
        if a.is_filler:
            return event(Decision.SUPPRESS, "filler or small talk; no retrievable content")

        # 2) End-of-utterance: everything non-filler is forwarded in full.
        if final:
            self._retrieved.update(stable)
            return event(
                Decision.PROVISIONAL_RETRIEVE,
                "end of utterance: final flush of complete transcript",
                new_keys,
            )

        # 3) Incomplete clause: keep listening.
        if a.is_dangling:
            return event(Decision.WAIT, "incomplete clause (dangling function word / open punctuation)")

        # 4) Stable entity cluster with something new in it -> retrieve.
        if len(stable) >= self._min_cluster and len(new_keys) >= self._min_new:
            self._retrieved.update(stable)
            clause = "complete" if a.has_predicate and a.ends_terminal else "open"
            return event(
                Decision.PROVISIONAL_RETRIEVE,
                f"stable entity cluster ({len(new_keys)} new, {len(stable)} total; clause {clause})",
                new_keys,
            )

        # 5) Otherwise nothing actionable yet.
        if not a.entities:
            return event(Decision.WAIT, "no entities detected yet")
        if not stable:
            return event(Decision.WAIT, "entities still unstable (trailing edge may be growing)")
        return event(Decision.WAIT, "no new stable entities since last retrieval")

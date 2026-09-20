"""
stage4_generator.py
===================

Stage 4 (Synthesis & Generation) of the Streaming Live RAG pipeline.

Takes the user's spoken `transcript` plus Stage 3's `topic_context`
(`{topic_id: [unique text facts]}`) and streams back a spoken-style answer,
token by token, ready to be fed into a Text-to-Speech engine.

    gen = Stage4Generator(client=shared_groq_client)
    async for token in gen.generate_stream(transcript, topic_context):
        ...

Design notes
------------
* The class is stateless: every call is independent, so one instance can be shared by
  all WebSocket connections. All per-session state lives in `StreamSession`.
* The model is `openai/gpt-oss-120b` on Groq's OpenAI-compatible endpoint. It is a
  reasoning model, so `reasoning_effort` defaults to "low" to keep time-to-first-token
  small for voice. Only `delta.content` is yielded; reasoning deltas are ignored.
* The system prompt forbids Markdown, but models slip. As defence in depth, every
  token is passed through a tiny filter that drops `* # ` ~` so nothing formatting-like
  can ever reach the TTS engine.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from typing import Any, AsyncGenerator, Optional

from openai import AsyncOpenAI

logger = logging.getLogger("stage4")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_STAGE4_MODEL = "openai/gpt-oss-120b"
NO_CONTEXT = "No context available."

SYSTEM_PROMPT = """You are a direct, natural voice assistant answering user queries using retrieved knowledge facts.

STRICT BEHAVIOR RULES:
1. EXPLICIT INTENT MATCHING: Answer ONLY the exact question asked in the user's current transcript.
2. PRE-FETCHED CONTEXT FILTERING: The provided <context> contains pre-fetched facts (cancellation policies, deposit terms, catering rates, equipment details). DO NOT mention any of these extra details unless the user explicitly asks about them in their transcript.
3. NO UNASKED WARNINGS OR HEADS-UPS: Do not add "Just a heads up on terms...", "Note on policies...", or "Prices start at..." unless asked.

--- FEW-SHOT EXAMPLES ---

EXAMPLE 1 (Do NOT volunteer policies):
[User Transcript]: "I need to book a conference hall at Marriott Pune for 50 people next Friday."
[Context]: 
- Topic 1: Marriott Pune offers 8 meeting rooms with capacities from 20 to 120 people.
- Topic 1: Cancellation policy: free up to 30 days, 50% charged within 15-29 days.
- Topic 1: Deposit: 25% advance required.
[Correct Assistant Response]: "Marriott Pune has eight meeting rooms that accommodate groups from 20 up to 120 people, which will fit a group of 50 nicely. The Grand Ballroom can also be divided into sections for your size. Would you like me to proceed with reserving a space?"
[Incorrect Assistant Response]: "...Also, just a heads-up, cancellation is free up to 30 days..." (WRONG - User did not ask about cancellation!)

EXAMPLE 2 (Answering explicit policy query):
[User Transcript]: "What are their cancellation policies and does it include a projector?"
[Context]:
- Topic 1: Cancellation policy: free up to 30 days, 50% charged within 15-29 days.
- Topic 1: Every meeting room includes a ceiling-mounted 4K projector.
[Correct Assistant Response]: "You can cancel free of charge up to 30 days before the event. Cancellations between 15 and 29 days out incur a 50% fee, and full charges apply within 14 days. As for equipment, every meeting room includes a 4K projector and screen at no extra cost."
--- END EXAMPLES ---
"""

# Characters that only ever mean "formatting" in Markdown; useless (and harmful) in TTS input.
_MARKDOWN_CHARS = re.compile(r"[*#`~]")


class Stage4Generator:
    """Async, streaming answer generator for Stage 4."""

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_STAGE4_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        reasoning_effort: Optional[str] = "low",
    ) -> None:
        """
        `client`: optionally reuse an existing `AsyncOpenAI` client that already points at
        Groq (the app shares one HTTP pool across stages). If omitted, one is created against
        `GROQ_BASE_URL` using `api_key` or the `GROQ_API_KEY` environment variable.

        `max_tokens` also bounds the model's hidden reasoning, so keep it generous.
        `reasoning_effort`: "low" | "medium" | "high"; falsy disables the parameter.
        """
        if client is None:
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("Stage4Generator needs a client or a GROQ_API_KEY")
            client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=key, timeout=30.0, max_retries=2)
            self._owns_client = True
        else:
            self._owns_client = False

        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        """Close the HTTP client, but only if this instance created it."""
        if self._owns_client:
            await self._client.close()

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_context(topic_context: dict[str, Any]) -> str:
        """
        Flatten `{topic_id: [facts]}` into one readable block of facts: one fact per line,
        a blank line between topics (so unrelated topics stay visually separate), and no topic ids,
        which are internal names that mean nothing to the model. Empty -> "No context available."
        """
        groups: list[str] = []
        for facts in (topic_context or {}).values():
            items = [facts] if isinstance(facts, str) else list(facts or [])
            lines = [str(f).strip() for f in items if str(f).strip()]
            if lines:
                groups.append("\n".join(lines))
        return "\n\n".join(groups) or NO_CONTEXT

    @classmethod
    def _build_user_message(cls, transcript: str, topic_context: dict[str, Any]) -> str:
        return (
            f"<context>\n{cls.format_context(topic_context)}\n</context>\n\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Reply to the user now, in spoken style, using only the information above."
        )

    # ------------------------------------------------------------------ #
    # Streaming generation
    # ------------------------------------------------------------------ #
    async def generate_stream(
        self,
        transcript: str,
        topic_context: dict[str, list[str]],
    ) -> AsyncGenerator[str, None]:
        """
        Yield speakable response text as it is generated.

        Yields nothing for a blank transcript. Raises `RuntimeError` if the model finishes
        without producing any text (e.g. the token budget was consumed by reasoning), and lets
        API errors from the OpenAI SDK propagate so the caller can report them.
        Closing this generator early (task cancellation) also closes the HTTP stream.
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return

        request: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_message(transcript, topic_context)},
            ],
            "stream": True,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._reasoning_effort:
            request["extra_body"] = {"reasoning_effort": self._reasoning_effort}

        stream = await self._client.chat.completions.create(**request)
        produced = False
        finish_reason: Optional[str] = None
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                piece = choice.delta.content
                if not piece:
                    continue  # role-only / reasoning-only deltas
                piece = _MARKDOWN_CHARS.sub("", piece)
                if piece:
                    produced = True
                    yield piece
        finally:
            with contextlib.suppress(Exception):
                await stream.close()

        if not produced:
            logger.warning("Stage 4 produced no text (finish_reason=%s)", finish_reason)
            raise RuntimeError(f"model returned no text (finish_reason={finish_reason})")
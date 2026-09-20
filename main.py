"""
main.py
=======

FastAPI entry point for the Streaming Live RAG pipeline (Stages 1 + 2 + 3 + 4 + 5).

    python seed_db.py            # once, to create ./chroma_db
    uvicorn main:app --reload    # or: python main.py

WebSocket endpoint: `/ws/stream`

Client -> server messages (JSON, or plain text which is treated as a chunk):
    {"type": "chunk", "text": "tell me about Marriott Pune"}
    {"type": "chunk", "text": "and the catering.", "final": true}   # chunk + end of utterance
    {"type": "end_of_utterance"}
    {"type": "generate_response"}   # Stage 4: answer current transcript + Stage 5 TTS stream
    {"type": "reset"}          # clear ALL session state
    {"type": "ping"}

Server -> client log events (all carry "type" and "ts"):
    ready              - session info
    chunk_received     - echo of current buffer
    stage1_decision    - WAIT | SUPPRESS | PROVISIONAL_RETRIEVE
    stage2_started     - decomposition began
    stage2_result      - sub_queries
    stage3_result      - new documents and topic context
    stage4_started     - generation began
    stage4_token       - streamed speakable text token
    audio_chunk        - base64 encoded audio from Stage 5 TTS
    stage4_done        - generation + TTS complete
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from stage1_controller import NLUClassifier, Stage1Controller, Stage1Event
from stage2_decomposer import (
    DEFAULT_MODEL,
    GROQ_BASE_URL,
    DecomposerError,
    Stage2Decomposer,
)
from stage3_retriever import DEFAULT_COLLECTION, DEFAULT_DB_PATH, Stage3Retriever
from stage4_generator import DEFAULT_STAGE4_MODEL, Stage4Generator
from stage5_tts import Stage5TTSPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("main")

DEBOUNCE_MS = int(os.getenv("DEBOUNCE_MS", "500"))
MAX_WAIT_MS = int(os.getenv("MAX_WAIT_MS", "2500"))
MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", DEFAULT_DB_PATH)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION)
STAGE3_TOP_K = int(os.getenv("STAGE3_TOP_K", "3"))
STAGE3_MAX_DISTANCE = float(os.getenv("STAGE3_MAX_DISTANCE", "0.8"))

STAGE4_MODEL = os.getenv("STAGE4_MODEL", DEFAULT_STAGE4_MODEL)
STAGE4_TEMPERATURE = float(os.getenv("STAGE4_TEMPERATURE", "0.3"))
STAGE4_MAX_TOKENS = int(os.getenv("STAGE4_MAX_TOKENS", "1024"))
STAGE4_REASONING_EFFORT = os.getenv("STAGE4_REASONING_EFFORT", "low")
STAGE4_SETTLE_TIMEOUT_S = int(os.getenv("STAGE4_SETTLE_TIMEOUT_MS", "5000")) / 1000

Emit = Callable[..., None]


def _snapshot_topic_context(ctx: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not isinstance(ctx, dict):
        return out

    for topic_id, val in ctx.items():
        tid = str(topic_id)
        out[tid] = []
        
        if isinstance(val, dict):
            raw_facts = val.get("facts", [])
        elif isinstance(val, list):
            raw_facts = val
        elif isinstance(val, str):
            raw_facts = [val]
        else:
            raw_facts = []

        for item in raw_facts:
            if isinstance(item, dict):
                text = item.get("text") or item.get("fact") or str(item)
            else:
                text = str(item)
            
            text_clean = text.strip()
            if text_clean and text_clean not in out[tid]:
                out[tid].append(text_clean)
                
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = await asyncio.to_thread(NLUClassifier)

    api_key = os.getenv("GROQ_API_KEY")
    app.state.llm_client = (
        AsyncOpenAI(
            base_url=GROQ_BASE_URL,
            api_key=api_key,
            timeout=30.0,
            max_retries=2,
            default_headers={
                "HTTP-Referer": os.getenv("GROQ_REFERER", "http://localhost:8000"),
                "X-Title": os.getenv("GROQ_APP_TITLE", "Streaming Live RAG"),
            },
        )
        if api_key
        else None
    )
    if app.state.llm_client is None:
        logger.warning("GROQ_API_KEY not set - Stages 2, 3 and 4 will be disabled.")

    app.state.stage3 = None
    app.state.stage3_error = None
    try:
        app.state.stage3 = await asyncio.to_thread(
            Stage3Retriever,
            CHROMA_DB_PATH,
            CHROMA_COLLECTION,
            top_k=STAGE3_TOP_K,
            max_distance=STAGE3_MAX_DISTANCE,
        )
    except Exception as exc:
        app.state.stage3_error = str(exc)
        logger.error("Stage 3 disabled: %s", exc)

    app.state.stage4 = (
        Stage4Generator(
            client=app.state.llm_client,
            model=STAGE4_MODEL,
            temperature=STAGE4_TEMPERATURE,
            max_tokens=STAGE4_MAX_TOKENS,
            reasoning_effort=STAGE4_REASONING_EFFORT or None,
        )
        if app.state.llm_client is not None
        else None
    )

    yield

    if app.state.stage4 is not None:
        await app.state.stage4.aclose()
    if app.state.llm_client is not None:
        await app.state.llm_client.close()


app = FastAPI(title="Streaming Live RAG - Stages 1-5", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "nlu_backend": app.state.classifier.backend,
        "stage2_enabled": app.state.llm_client is not None,
        "stage3_enabled": app.state.stage3 is not None,
        "stage3_error": app.state.stage3_error,
        "stage4_enabled": app.state.stage4 is not None,
        "stage5_enabled": True,
        "model": MODEL,
        "stage4_model": STAGE4_MODEL,
        "chroma_path": CHROMA_DB_PATH,
        "collection": CHROMA_COLLECTION,
    }


class PipelineWorker:
    def __init__(
        self,
        decomposer: Stage2Decomposer,
        retriever: Optional[Stage3Retriever],
        emit: Emit,
        on_topic_context: Optional[Callable[[dict[str, list[str]]], None]] = None,
    ) -> None:
        self._decomposer = decomposer
        self._retriever = retriever
        self._emit = emit
        self._on_topic_context = on_topic_context
        self._queue: deque[Stage1Event] = deque()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self._task = asyncio.create_task(self._run(), name="pipeline-worker")

    def submit(self, event: Stage1Event) -> None:
        if self._closed:
            return
        self._idle.clear()
        if self._queue:
            last = self._queue[-1]
            if last.utterance_id == event.utterance_id and not last.final:
                self._queue[-1] = event
                self._emit(
                    "stage2_coalesced",
                    utterance_id=event.utterance_id,
                    dropped_transcript=last.buffer,
                    replaced_by=event.buffer,
                )
                self._wake.set()
                return
        self._queue.append(event)
        self._wake.set()

    async def wait_idle(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def aclose(self) -> None:
        self._closed = True
        self._queue.clear()
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._idle.set()

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            while self._queue:
                await self._process(self._queue.popleft())
            self._wake.clear()
            self._idle.set()

    async def _process(self, ev: Stage1Event) -> None:
        base = {"utterance_id": ev.utterance_id, "final": ev.final, "transcript": ev.buffer}

        self._emit("stage2_started", trigger=ev.trigger, **base)
        try:
            result = await self._decomposer.decompose(ev.buffer, final=ev.final)
        except DecomposerError as exc:
            logger.warning("Stage 2 failed: %s", exc)
            self._emit("stage2_error", message=str(exc), **base)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Stage 2 unexpected failure")
            self._emit("stage2_error", message=f"unexpected: {exc!r}", **base)
            return

        self._emit("stage2_result", known_topics=self._decomposer.topics, **base, **result.to_dict())

        if self._retriever is None:
            return

        sub_queries = [sq.model_dump() for sq in result.sub_queries]
        if not sub_queries:
            self._emit("stage3_skipped", reason="Stage 2 produced no new sub-queries", **base)
            return
        try:
            s3 = await asyncio.to_thread(self._retriever.retrieve, sub_queries)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Stage 3 failure")
            self._emit("stage3_error", message=str(exc), **base)
            return
        s3_payload = s3.to_dict()
        self._emit("stage3_result", **base, **s3_payload)
        if self._on_topic_context is not None and isinstance(s3_payload.get("topic_context"), dict):
            self._on_topic_context(_snapshot_topic_context(s3_payload["topic_context"]))


class StreamSession:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._out: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sender: Optional[asyncio.Task] = None
        self._stage1: Optional[Stage1Controller] = None
        self._decomposer: Optional[Stage2Decomposer] = None
        self._retriever: Optional[Stage3Retriever] = None
        self._worker: Optional[PipelineWorker] = None
        self._generator: Optional[Stage4Generator] = None
        self._tts: Optional[Stage5TTSPipeline] = None
        self._gen_task: Optional[asyncio.Task] = None
        self._response_seq = 0
        self._active_response_id: Optional[int] = None
        self._utterance_open = False
        self._last_transcript = ""
        self._last_topic_context: dict[str, list[str]] = {}

    def emit(self, type_: str, **data: Any) -> None:
        self._out.put_nowait({"type": type_, "ts": time.time(), **data})

    async def run(self) -> None:
        await self.ws.accept()
        state = self.ws.app.state
        self._sender = asyncio.create_task(self._send_loop(), name="ws-sender")

        if state.llm_client is not None:
            self._decomposer = Stage2Decomposer(client=state.llm_client, model=MODEL)
            if state.stage3 is not None:
                self._retriever = state.stage3.fork()
            else:
                self.emit("warning", message=f"Stage 3 disabled: {state.stage3_error}")
            self._generator = state.stage4
            self._tts = Stage5TTSPipeline(voice="en-US-SteffanNeural")
            self._start_worker()
        else:
            self.emit("warning", message="GROQ_API_KEY not set; Stages 2, 3, 4 and 5 disabled.")

        self._stage1 = Stage1Controller(
            on_event=self._on_stage1_event,
            debounce_ms=DEBOUNCE_MS,
            max_wait_ms=MAX_WAIT_MS,
            classifier=state.classifier,
        )
        self.emit(
            "ready",
            nlu_backend=self._stage1.backend,
            model=MODEL,
            stage4_model=STAGE4_MODEL,
            stage2_enabled=self._decomposer is not None,
            stage3_enabled=self._retriever is not None,
            stage4_enabled=self._generator is not None,
            stage5_enabled=self._tts is not None,
            debounce_ms=DEBOUNCE_MS,
            max_wait_ms=MAX_WAIT_MS,
        )

        try:
            while True:
                await self._handle(await self.ws.receive_text())
        except WebSocketDisconnect:
            logger.info("client disconnected")
        except Exception:
            logger.exception("session crashed")
        finally:
            await self._shutdown()

    def _start_worker(self) -> None:
        assert self._decomposer is not None
        self._worker = PipelineWorker(
            self._decomposer, self._retriever, self.emit, on_topic_context=self._remember_topic_context
        )

    def _remember_topic_context(self, snapshot: dict[str, list[str]]) -> None:
        for tid, fact_list in snapshot.items():
            if tid not in self._last_topic_context:
                self._last_topic_context[tid] = []
            for f in fact_list:
                if f not in self._last_topic_context[tid]:
                    self._last_topic_context[tid].append(f)

    async def _stop_and_clear_stages_2_3(self) -> None:
        if self._worker is not None:
            await self._worker.aclose()
            self._worker = None
        if self._decomposer is not None:
            self._decomposer.reset()
        if self._retriever is not None:
            self._retriever.reset()
        self._last_topic_context = {}

    async def _reset_pipeline(self) -> None:
        await self._cancel_generation("reset")
        if self._stage1 is not None:
            self._stage1.reset()
        self._utterance_open = False
        self._last_transcript = ""
        await self._stop_and_clear_stages_2_3()
        if self._decomposer is not None:
            self._start_worker()

    async def _shutdown(self) -> None:
        await self._cancel_generation("disconnect")
        if self._stage1 is not None:
            await self._stage1.close()
            self._stage1.reset()
        self._utterance_open = False
        self._last_transcript = ""
        await self._stop_and_clear_stages_2_3()
        if self._sender is not None:
            self._sender.cancel()
            await asyncio.gather(self._sender, return_exceptions=True)

    async def _on_stage1_event(self, ev: Stage1Event) -> None:
        self.emit("stage1_decision", **ev.to_dict())
        if ev.buffer:
            self._last_transcript = ev.buffer
        if ev.final:
            self._utterance_open = False
        if ev.should_decompose and self._worker is not None:
            self._worker.submit(ev)

    def _current_transcript(self) -> str:
        live = self._stage1.buffer if self._stage1 is not None else ""
        return (live or self._last_transcript or "").strip()

    def _current_topic_context(self) -> dict[str, list[str]]:
        ctx: dict[str, list[str]] = {k: list(v) for k, v in self._last_topic_context.items()}

        if self._retriever is not None:
            retriever_raw = None
            for name in ("topic_context", "_topic_context", "get_topic_context"):
                candidate = getattr(self._retriever, name, None)
                if callable(candidate):
                    candidate = candidate()
                if isinstance(candidate, dict):
                    retriever_raw = candidate
                    break

            if retriever_raw:
                live_snapshot = _snapshot_topic_context(retriever_raw)
                for tid, fact_list in live_snapshot.items():
                    if tid not in ctx:
                        ctx[tid] = []
                    for f in fact_list:
                        if f not in ctx[tid]:
                            ctx[tid].append(f)

        return ctx
    
    async def _start_generation(self) -> None:
        if self._generator is None:
            self.emit("stage4_error", message="Stage 4 disabled: GROQ_API_KEY not set")
            return
        await self._cancel_generation("superseded")
        if self._utterance_open and self._stage1 is not None:
            await self._end_utterance()
        self._response_seq += 1
        self._active_response_id = self._response_seq
        self._gen_task = asyncio.create_task(
            self._run_generation(self._response_seq), name=f"stage4-{self._response_seq}"
        )

    async def _end_utterance(self) -> None:
        assert self._stage1 is not None
        self._utterance_open = False
        await self._stage1.end_utterance()

    async def _cancel_generation(self, reason: str) -> None:
        task, self._gen_task = self._gen_task, None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.emit("stage4_cancelled", response_id=self._active_response_id, reason=reason)

    async def _run_generation(self, response_id: int) -> None:
        assert self._generator is not None
        started = time.perf_counter()
        try:
            await asyncio.sleep(0.1)

            if self._worker is not None and not await self._worker.wait_idle(STAGE4_SETTLE_TIMEOUT_S):
                self.emit("warning", message="Stage 2/3 still busy; generating from context gathered so far")

            transcript = self._current_transcript()
            topic_context = self._current_topic_context()
            if not transcript:
                self.emit("stage4_error", response_id=response_id, message="no transcript to respond to yet")
                return

            facts = sum(len(v) for v in topic_context.values())
            self.emit(
                "stage4_started",
                response_id=response_id,
                transcript=transcript,
                topics=len(topic_context),
                facts=facts,
            )

            tokens = 0
            first_token_ms: Optional[int] = None

            # Helper generator that emits text tokens to the client and feeds Stage 5 TTS
            async def token_stream():
                nonlocal tokens, first_token_ms
                async for token in self._generator.generate_stream(transcript, topic_context):
                    if first_token_ms is None:
                        first_token_ms = int((time.perf_counter() - started) * 1000)
                    tokens += 1
                    self.emit("stage4_token", response_id=response_id, token=token)
                    yield token

            # Pipeline text tokens into Stage 5 TTS and emit audio chunks
            if self._tts is not None:
                async for audio_payload in self._tts.stream_text_to_audio(token_stream()):
                    self.emit(
                        "audio_chunk",
                        response_id=response_id,
                        text=audio_payload.get("text", ""),
                        audio_b64=audio_payload.get("audio_b64", ""),
                    )
            else:
                async for _ in token_stream():
                    pass

            latency_ms = int((time.perf_counter() - started) * 1000)
            self.emit(
                "stage4_done",
                response_id=response_id,
                tokens=tokens,
                first_token_ms=first_token_ms,
                latency_ms=latency_ms,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Stage 4/5 failure")
            self.emit("stage4_error", response_id=response_id, message=str(exc))

    async def _handle(self, raw: str) -> None:
        assert self._stage1 is not None
        msg: Any = raw
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            pass

        if isinstance(msg, str):
            msg = {"type": "chunk", "text": msg}
        if not isinstance(msg, dict):
            self.emit("warning", message="expected a JSON object or plain text")
            return

        kind = msg.get("type", "chunk" if "text" in msg else None)
        if kind == "chunk":
            text = msg.get("text")
            if not isinstance(text, str):
                self.emit("warning", message='"chunk" messages need a string "text" field')
                return
            self._stage1.feed(text)
            self._utterance_open = True
            if self._stage1.buffer:
                self._last_transcript = self._stage1.buffer
            self.emit("chunk_received", chunk=text, buffer=self._stage1.buffer, utterance_id=self._stage1.utterance_id)
            if msg.get("final"):
                await self._end_utterance()
        elif kind in ("end_of_utterance", "eou"):
            await self._end_utterance()
        elif kind == "generate_response":
            await self._start_generation()
        elif kind == "reset":
            await self._reset_pipeline()
            self.emit("reset_ack")
        elif kind == "ping":
            self.emit("pong")
        else:
            self.emit("warning", message=f"unknown message type: {kind!r}")

    async def _send_loop(self) -> None:
        try:
            while True:
                await self.ws.send_json(await self._out.get())
        except Exception:
            return


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await StreamSession(ws).run()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
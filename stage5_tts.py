import asyncio
import base64
import re
import edge_tts

# Punctuation regex to detect natural pauses for speech synthesis
CLAUSE_BOUNDARIES = re.compile(r'(?<=[.,!?;:\n])\s+')


class Stage5TTSPipeline:

    def __init__(self, voice: str = "en-US-SteffanNeural"):
        self.voice = voice

    async def stream_text_to_audio(self, token_generator):
        """Buffers text tokens from Stage 4 and yields base64 audio payloads

        whenever a complete sentence or clause is ready.
        """
        buffer = ""

        async for token in token_generator:
            buffer += token

            # Split buffer into clauses at punctuation marks
            clauses = CLAUSE_BOUNDARIES.split(buffer)

            # Synthesize all completed clauses
            while len(clauses) > 1:
                clause_to_speak = clauses.pop(0).strip()
                if clause_to_speak:
                    audio_b64 = await self._synthesize_to_b64(clause_to_speak)
                    if audio_b64:
                        yield {
                            "type": "audio_chunk",
                            "text": clause_to_speak,
                            "audio_b64": audio_b64,
                        }

            buffer = clauses[0] if clauses else ""

        # Flush any remaining text buffer at the end of generation
        if buffer.strip():
            audio_b64 = await self._synthesize_to_b64(buffer.strip())
            if audio_b64:
                yield {
                    "type": "audio_chunk",
                    "text": buffer.strip(),
                    "audio_b64": audio_b64,
                }

    async def _synthesize_to_b64(self, text: str) -> str:
        """Helper to call Edge-TTS and convert raw MP3 bytes to base64."""
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            audio_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.extend(chunk["data"])

            return base64.b64encode(bytes(audio_bytes)).decode("utf-8")
        except Exception as e:
            print(f"\n[Stage 5 TTS Error]: {e}")
            return ""
import asyncio
import base64
import json
import os
import tempfile
import websockets

# Suppress pygame banner message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

WS_URL = "ws://127.0.0.1:8000/ws/stream"

# Track pending audio tasks so main() waits for playback to finish
audio_tasks = set()


async def play_audio_chunk(audio_b64: str):
    """Decodes Base64 audio chunk and plays it via Pygame."""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        print(f" [🔊 Playing Audio: {len(audio_bytes)} bytes]", end="", flush=True)

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Temporary file for reliable MP3 decoding on Windows
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            sound = pygame.mixer.Sound(tmp_path)
            channel = sound.play()
            while channel and channel.get_busy():
                await asyncio.sleep(0.05)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    except Exception as e:
        print(f"\n[❌ Audio Playback Error]: {e}")


async def listen_loop(websocket, done_event: asyncio.Event):
    """Listens for server messages, prints tokens, and plays audio chunks."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                # Stream live response text
                if msg_type == "stage4_token":
                    print(data.get("token", ""), end="", flush=True)

                # Play incoming TTS audio chunks asynchronously
                elif msg_type == "audio_chunk":
                    audio_b64 = data.get("audio_b64")
                    if audio_b64:
                        task = asyncio.create_task(play_audio_chunk(audio_b64))
                        audio_tasks.add(task)
                        task.add_done_callback(audio_tasks.discard)

                elif msg_type == "stage4_done":
                    print("\n\n--- Generation Complete ---")
                    done_event.set()

                elif msg_type == "warning":
                    print(f"\n[Warning]: {data.get('message')}")

                elif msg_type == "stage4_error":
                    print(f"\n[Error]: {data.get('message')}")
                    done_event.set()

            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        done_event.set()


async def run_test_case_1(websocket):
    print("\n========================================================")
    print(" Running Test Case 1: Multi-Utterance State Accumulation")
    print("========================================================\n")

    # Distinct conversational utterances sent with natural delays
    utterances = [
        "Hi, I'm planning an executive workshop at Marriott Pune.",
        "Could you list the seating capacities for their meeting rooms?",
        "Also, what amenities and AV equipment do they provide?",
        "Make sure to include details for a group of around 40 people."
    ]

    for i, utterance in enumerate(utterances, start=1):
        print(f"-> Sending Utterance {i}: '{utterance}'")
        await websocket.send(json.dumps({"type": "chunk", "text": utterance + " "}))
        # Simulate natural speech pauses between distinct utterances
        await asyncio.sleep(1.0)

    print("\n-> Requesting response generation...")
    await websocket.send(json.dumps({"type": "generate_response"}))


async def run_test_case_2(websocket):
    print("\n========================================================")
    print(" Running Test Case 2: In-Stream Topic Switch")
    print("========================================================\n")

    chunks = [
        "I need to find a venue in Pune ",
        "for 30 people, ",
        "and oh wait, I also need ",
        "to check flight prices from Delhi ",
        "to Pune for next Thursday.",
    ]

    for chunk in chunks:
        print(f"-> Sending chunk: '{chunk}'")
        await websocket.send(json.dumps({"type": "chunk", "text": chunk}))
        await asyncio.sleep(0.3)

    print("\n-> Requesting response generation...")
    await websocket.send(json.dumps({"type": "generate_response"}))


async def run_test_case_3(websocket):
    print("\n========================================================")
    print(" Running Test Case 3: Single-Burst Utterance (No Input Chunking)")
    print("========================================================\n")

    full_query = (
        "I need a venue in Pune for 30 people next Thursday, and also check flight prices from Delhi to Pune for the same day."
    )

    print(f"-> Sending FULL query in a single burst:\n   '{full_query}'\n")

    await websocket.send(
        json.dumps({"type": "chunk", "text": full_query, "final": True})
    )

    print("-> Requesting response generation...")
    await websocket.send(json.dumps({"type": "generate_response"}))


async def main():
    print("Select a Test Case to Run:")
    print("  [1] Multi-Utterance State (Pune Marriott Capacities & Amenities)")
    print("  [2] In-Stream Topic Switch (Pune Venues & Flights)")
    print("  [3] Single-Burst Utterance (No Input Chunking)")

    choice = input("\nEnter choice (1-3) [Default: 2]: ").strip() or "2"

    done_event = asyncio.Event()

    async with websockets.connect(WS_URL) as websocket:
        listener_task = asyncio.create_task(listen_loop(websocket, done_event))

        await asyncio.sleep(0.5)

        if choice == "1":
            await run_test_case_1(websocket)
        elif choice == "2":
            await run_test_case_2(websocket)
        else:
            await run_test_case_3(websocket)

        # Wait until server finishes sending generation
        await done_event.wait()

        # Wait for remaining audio chunks to finish playing through speakers
        if audio_tasks:
            print("\n[Waiting for remaining audio playback to finish...]")
            await asyncio.gather(*list(audio_tasks), return_exceptions=True)

        listener_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
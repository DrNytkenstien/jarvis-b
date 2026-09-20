"""
seed_db.py
==========

Populates a local ChromaDB collection (``streaming_rag_docs``) with sample facts about
Pune venues (Marriott Pune), amenities, catering, cancellation policies and Delhi -> Pune
flights, so the Streaming Live RAG pipeline has something to retrieve.

    python seed_db.py            # create / update (idempotent - uses upsert)
    python seed_db.py --reset    # drop the collection first (use after changing embedders)

Embeddings: ChromaDB's ONNX ``all-MiniLM-L6-v2`` (``onnxruntime``, no PyTorch). The model
(~80 MB) is downloaded to ``~/.cache/chroma/onnx_models`` on first run.

NOTE: every fact below is *illustrative sample data* (invented prices, times and policies).
Replace ``DOCS`` with your real content before relying on any answer.

Run this from the same working directory as the server (or pass the same --path), because
the default database location is the relative path ``./chroma_db``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "streaming_rag_docs"

# (id, category, entity, text)
DOCS: list[tuple[str, str, str, str]] = [
    # ---- Marriott Pune: venues & rooms ------------------------------------------------
    ("venue-001", "venue", "Marriott Pune",
     "Marriott Pune has a Grand Ballroom that seats 600 guests theatre-style or 400 banquet-style, "
     "and it can be divided into three independent sections."),
    ("venue-002", "venue", "Marriott Pune",
     "Marriott Pune offers 8 meeting rooms with capacities from 20 to 120 people, suitable for "
     "workshops, breakout sessions and corporate conferences."),
    ("venue-003", "venue", "Marriott Pune",
     "The Executive Boardroom at Marriott Pune seats 14 people around a single table and includes "
     "a video-conferencing setup and a private pre-function lounge."),
    ("venue-004", "venue", "Marriott Pune",
     "Marriott Pune has an outdoor lawn that hosts up to 250 guests for evening receptions and "
     "team-building events, weather permitting."),
    ("venue-005", "venue", "Marriott Pune",
     "Marriott Pune's event spaces are step-free and wheelchair accessible, with accessible "
     "restrooms on the same floor as the meeting rooms."),
    ("rooms-001", "rooms", "Marriott Pune",
     "Marriott Pune offers a corporate group rate starting at INR 7,500 per night including "
     "breakfast for room blocks of 10 or more rooms."),
    ("rooms-002", "rooms", "Marriott Pune",
     "Check-in at Marriott Pune is from 3 PM and check-out is at 12 noon. Early check-in and late "
     "check-out are subject to availability."),
    # ---- Amenities: projectors, AV, Wi-Fi, parking ---------------------------------------
    ("amenity-001", "amenities", "Marriott Pune",
     "Every meeting room at Marriott Pune includes a ceiling-mounted 4K projector and a projection "
     "screen at no extra charge. HDMI and wireless casting are both supported."),
    ("amenity-002", "amenities", "Marriott Pune",
     "The Marriott Pune Grand Ballroom has three projectors and a stage. An LED video wall can be "
     "added for INR 45,000 per day."),
    ("amenity-003", "amenities", "Marriott Pune",
     "Marriott Pune provides wireless and lapel microphones, a sound system and podium. An on-site "
     "AV technician is complimentary for events over 100 guests, otherwise INR 8,000 per day."),
    ("amenity-004", "amenities", "Marriott Pune",
     "Basic Wi-Fi is complimentary in all Marriott Pune event spaces. Dedicated 500 Mbps event "
     "bandwidth is available for INR 15,000 per day."),
    ("amenity-005", "amenities", "Marriott Pune",
     "Marriott Pune offers complimentary valet parking for up to 150 vehicles for event attendees."),
    ("amenity-006", "amenities", "Marriott Pune",
     "Marriott Pune can arrange an airport shuttle from Pune Airport (PNQ) at INR 1,800 per car; the "
     "drive takes roughly 30 to 45 minutes depending on traffic."),
    # ---- Catering ------------------------------------------------------------------------
    ("catering-001", "catering", "Marriott Pune",
     "Marriott Pune conference catering packages: full-day at INR 2,200 per person including two "
     "tea breaks and a buffet lunch; half-day at INR 1,400 per person including one tea break and "
     "a light lunch."),
    ("catering-002", "catering", "Marriott Pune",
     "Marriott Pune gala dinner buffet is INR 3,800 per person with a minimum of 50 guests, "
     "including live counters and desserts."),
    ("catering-003", "catering", "Marriott Pune",
     "Catering menus at Marriott Pune cover vegetarian, vegan, Jain and halal diets. Special "
     "dietary requests need 72 hours notice."),
    ("catering-004", "catering", "Marriott Pune",
     "Only in-house catering is allowed at Marriott Pune; outside caterers are not permitted, and "
     "external cakes attract a corkage fee."),
    # ---- Cancellation & payment policies ------------------------------------------------
    ("policy-001", "cancellation", "Marriott Pune",
     "Marriott Pune venue booking cancellation policy: free cancellation up to 30 days before the "
     "event; 50% of the venue rental is charged for cancellations 15 to 29 days before; 100% is "
     "charged within 14 days of the event."),
    ("policy-002", "cancellation", "Marriott Pune",
     "Room block cancellation at Marriott Pune: rooms can be released without penalty up to 14 days "
     "before arrival, with a 10% attrition allowance on the block."),
    ("policy-003", "cancellation", "Marriott Pune",
     "Catering guarantee at Marriott Pune: the final guest count is due 72 hours before the event. "
     "Reductions after that point are still billed at the guaranteed count."),
    ("policy-004", "cancellation", "Marriott Pune",
     "Marriott Pune allows one free date change if requested at least 21 days before the event, "
     "subject to availability of the venue on the new date."),
    ("policy-005", "payment", "Marriott Pune",
     "A 25% advance deposit secures a Marriott Pune event booking; the balance is due 7 days before "
     "the event. The deposit is refunded according to the cancellation schedule."),
    # ---- Flights: Delhi -> Pune ----------------------------------------------------------
    ("flight-001", "flights", "Delhi-Pune",
     "IndiGo operates several daily nonstop flights from Delhi (DEL) to Pune (PNQ), including "
     "departures at 06:05 and 08:40. Flight time is about 2 hours 10 minutes, with fares from "
     "INR 4,800."),
    ("flight-002", "flights", "Delhi-Pune",
     "Air India flies nonstop from Delhi to Pune twice daily, at 07:15 and 18:30, with fares from "
     "INR 5,600 including a meal service."),
    ("flight-003", "flights", "Delhi-Pune",
     "Akasa Air has one daily nonstop flight from Delhi to Pune departing at 21:10, with fares from "
     "INR 4,200. Meals are available for purchase on board."),
    ("flight-004", "flights", "Delhi-Pune",
     "Group bookings of 10 or more passengers on Delhi to Pune flights qualify for group fares, a "
     "7-day seat hold, and free name changes up to 72 hours before departure."),
    ("flight-005", "flights", "Delhi-Pune",
     "Domestic economy baggage allowance on most Delhi to Pune flights is 15 kg checked baggage "
     "and 7 kg cabin baggage per passenger."),
    ("flight-006", "flights", "Delhi-Pune",
     "Travelling from Delhi to Pune by train takes around 24 hours, so flying is recommended for "
     "corporate groups with tight schedules."),
    ("flight-007", "flights", "Mumbai-Pune",
     "There are no direct scheduled flights between Mumbai and Pune; the two cities are usually "
     "connected by road or rail in 3 to 4 hours."),
    # ---- General ------------------------------------------------------------------------
    ("city-001", "general", "Pune",
     "October to February is the most popular season for conferences in Pune thanks to pleasant "
     "weather; venues book out early, so reserve at least 8 weeks ahead."),
]


def get_embedding_function():
    """ONNX all-MiniLM-L6-v2 via onnxruntime - no PyTorch needed."""
    return ONNXMiniLM_L6_V2()


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the streaming_rag_docs ChromaDB collection.")
    ap.add_argument("--path", default=DEFAULT_DB_PATH, help="ChromaDB directory (default ./chroma_db)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--reset", action="store_true", help="delete the collection before seeding")
    ap.add_argument("--no-verify", action="store_true", help="skip the sample-query check")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # avoid cp1252 crashes on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading ONNX embedding model (first run downloads ~80 MB)...")
    ef = get_embedding_function()
    try:
        ef(["warm up"])
    except Exception as exc:
        print(f"ERROR: could not load the ONNX embedding model: {exc}\n"
              "Check your internet connection / proxy (the model is fetched on first use).")
        return 1

    client = chromadb.PersistentClient(path=args.path, settings=Settings(anonymized_telemetry=False))
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection '{args.collection}'.")
        except Exception:
            pass  # didn't exist

    try:
        collection = client.get_or_create_collection(
            name=args.collection,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},  # explicit: older Chroma versions default to L2
        )
    except Exception as exc:
        print(f"ERROR: could not open collection: {exc}\n"
              "If it was created with a different embedding function, re-run with --reset.")
        return 1

    collection.upsert(
        ids=[d[0] for d in DOCS],
        documents=[d[3] for d in DOCS],
        metadatas=[{"category": d[1], "entity": d[2], "source": "sample_data"} for d in DOCS],
    )
    print(f"Seeded '{args.collection}' at {args.path}: {collection.count()} documents.")

    if not args.no_verify:
        print("\nSanity check (cosine distance: lower = closer; use it to tune STAGE3_MAX_DISTANCE):")
        probes = [
            "Does the Marriott Pune have projectors?",
            "What happens if we cancel the event?",
            "Flights from Delhi to Pune",
            "What is the catering price per person?",
            "Tell me a joke about cats",  # should look clearly less relevant
        ]
        embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in ef(probes)]
        res = collection.query(query_embeddings=embeddings, n_results=2, include=["documents", "distances"])
        for i, q in enumerate(probes):
            print(f"\n  Q: {q}")
            for doc_id, dist, text in zip(res["ids"][i], res["distances"][i], res["documents"][i]):
                snippet = textwrap.shorten(text, width=90, placeholder="...")
                print(f"     {dist:5.3f}  {doc_id:<13} {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

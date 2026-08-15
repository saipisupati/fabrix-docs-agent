"""
kb/store.py — persist KB JSON + embeddings; retrieve by cosine similarity.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from functools import lru_cache
from typing import Any

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMBED_BATCH_SIZE, EMBEDDING_MODEL, EMBEDDINGS_URL, QDRANT_DIR
from kb.schema import KnowledgeBase

logger = logging.getLogger(__name__)

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kb")
KB_JSON_PATH = os.path.join(KB_DIR, "kb.json")
KB_EMBED_PATH = os.path.join(KB_DIR, "embeddings.npz")
KB_COLLECTION = "fabrix_kb"

# Split connect vs read so a black-holed TCP socket fails fast (cycle 23).
# Query embeds stay short; ingest batches keep a longer read budget.
EMBED_CONNECT_TIMEOUT_S = 5.0
EMBED_READ_TIMEOUT_S = 20.0
EMBED_BATCH_READ_TIMEOUT_S = 90.0


def ensure_kb_dir() -> None:
    os.makedirs(KB_DIR, exist_ok=True)


def save_kb(kb: KnowledgeBase, path: str = KB_JSON_PATH) -> None:
    ensure_kb_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb.to_dict(), f, indent=2, ensure_ascii=False)
    clear_kb_disk_cache()


@lru_cache(maxsize=4)
def _load_kb_cached(path: str) -> KnowledgeBase | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return KnowledgeBase.from_dict(json.load(f))


def load_kb(path: str = KB_JSON_PATH) -> KnowledgeBase | None:
    """Load KB JSON once per process (memoized)."""
    return _load_kb_cached(path)


def _embed_batch(
    texts: list[str],
    model: str,
    max_retries: int = 3,
    *,
    read_timeout: float | None = None,
) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }
    if read_timeout is None:
        read_timeout = (
            EMBED_BATCH_READ_TIMEOUT_S if len(texts) > 1 else EMBED_READ_TIMEOUT_S
        )
    timeout = (EMBED_CONNECT_TIMEOUT_S, read_timeout)
    preview = (texts[0][:80] + "…") if texts and len(texts[0]) > 80 else (texts[0] if texts else "")
    last_err = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            response = requests.post(
                EMBEDDINGS_URL,
                headers=headers,
                json={"model": model, "input": texts},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
            data.sort(key=lambda d: d["index"])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "embed ok attempt=%s/%s elapsed_ms=%.0f n_texts=%s q=%r",
                attempt + 1,
                max_retries,
                elapsed_ms,
                len(texts),
                preview,
            )
            return [d["embedding"] for d in data]
        except Exception as e:
            last_err = e
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "embed fail attempt=%s/%s elapsed_ms=%.0f n_texts=%s q=%r err=%s",
                attempt + 1,
                max_retries,
                elapsed_ms,
                len(texts),
                preview,
                type(e).__name__,
            )
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
                continue
            raise
    raise last_err  # pragma: no cover


@lru_cache(maxsize=256)
def embed_query(text: str, model: str = "") -> list[float]:
    model = model or EMBEDDING_MODEL
    return _embed_batch([text], model)[0]


def embed_entries(entries: list[dict[str, Any]], model: str = "") -> np.ndarray:
    model = model or EMBEDDING_MODEL
    texts = [e["text"][:6000] for e in entries]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        print(f"  Embedding KB batch {i // EMBED_BATCH_SIZE + 1} ({len(batch)} entries)...")
        vectors.extend(
            _embed_batch(batch, model, read_timeout=EMBED_BATCH_READ_TIMEOUT_S)
        )
    return np.array(vectors, dtype=np.float32)


def save_embeddings(entries: list[dict[str, Any]], matrix: np.ndarray, path: str = KB_EMBED_PATH) -> None:
    ensure_kb_dir()
    ids = np.array([e["id"] for e in entries])
    np.savez_compressed(path, vectors=matrix, ids=ids)
    clear_kb_disk_cache()


@lru_cache(maxsize=4)
def _load_embeddings_cached(path: str) -> tuple[np.ndarray, tuple[str, ...]] | None:
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=True)
    ids = tuple(str(x) for x in data["ids"].tolist())
    return data["vectors"].astype(np.float32), ids


def load_embeddings(path: str = KB_EMBED_PATH) -> tuple[np.ndarray, list[str]] | None:
    """Load embeddings.npz once per process (memoized)."""
    loaded = _load_embeddings_cached(path)
    if loaded is None:
        return None
    matrix, ids = loaded
    return matrix, list(ids)


def clear_kb_disk_cache() -> None:
    """Drop memoized KB / embeddings after rebuild or save."""
    _load_kb_cached.cache_clear()
    _load_embeddings_cached.cache_clear()


def upsert_qdrant_kb(entries: list[dict[str, Any]], matrix: np.ndarray, model: str) -> bool:
    """Best-effort write to Qdrant collection fabrix_kb (requires exclusive DB access)."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError:
        return False

    try:
        client = QdrantClient(path=QDRANT_DIR)
    except Exception as e:
        print(f"  Skipping Qdrant KB upsert (DB locked or unavailable): {e}")
        return False

    try:
        if client.collection_exists(KB_COLLECTION):
            client.delete_collection(KB_COLLECTION)
        client.create_collection(
            collection_name=KB_COLLECTION,
            vectors_config=VectorParams(size=len(matrix[0]), distance=Distance.COSINE),
        )
        points = []
        for i, (entry, vec) in enumerate(zip(entries, matrix)):
            points.append(
                PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={k: v for k, v in entry.items()},
                )
            )
        batch = 200
        for i in range(0, len(points), batch):
            client.upsert(collection_name=KB_COLLECTION, points=points[i : i + batch])
            print(f"  Uploaded KB points {min(i + batch, len(points))}/{len(points)}")
        # stamp model next to qdrant for debugging
        with open(os.path.join(QDRANT_DIR, "kb_embedding_model.txt"), "w", encoding="utf-8") as f:
            f.write(model)
        return True
    except Exception as e:
        print(f"  Qdrant KB upsert failed: {e}")
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass


def retrieve_kb(question: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Retrieve top KB entries using local embeddings.npz (preferred) or empty list."""
    kb = load_kb()
    if kb is None:
        return []
    entries = kb.searchable_entries()
    if not entries:
        return []

    loaded = load_embeddings()
    if loaded is None:
        return []
    matrix, ids = loaded
    id_to_entry = {e["id"]: e for e in entries}
    # Align matrix rows to current entries by id order in npz
    ordered = [id_to_entry[i] for i in ids if i in id_to_entry]
    if len(ordered) != len(matrix):
        # rebuild alignment: only use overlapping ids
        idx = [n for n, i in enumerate(ids) if i in id_to_entry]
        matrix = matrix[idx]
        ordered = [id_to_entry[ids[n]] for n in idx]
    if not ordered:
        return []

    from freshness import source_is_retired, retired_set

    retired = retired_set()
    if retired:
        keep_idx = [
            n
            for n, e in enumerate(ordered)
            if not source_is_retired(str(e.get("source") or ""), retired)
        ]
        if not keep_idx:
            return []
        matrix = matrix[keep_idx]
        ordered = [ordered[n] for n in keep_idx]

    model = kb.embedding_model or EMBEDDING_MODEL
    q = np.array(embed_query(question, model), dtype=np.float32)
    # cosine similarity
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) + 1e-9)
    scores = (matrix @ q) / (norms + 1e-9)
    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for i in top_idx:
        entry = dict(ordered[int(i)])
        entry["score"] = float(scores[int(i)])
        results.append(entry)
    return results

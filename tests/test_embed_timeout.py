"""Cycle 23: embed timeout split + KB disk memoization."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_embed_timeout_is_connect_read_tuple():
    from kb import store
    from query_qdrant import EMBED_CONNECT_TIMEOUT_S, EMBED_READ_TIMEOUT_S

    assert store.EMBED_CONNECT_TIMEOUT_S == 5.0
    assert store.EMBED_READ_TIMEOUT_S == 20.0
    assert store.EMBED_BATCH_READ_TIMEOUT_S >= store.EMBED_READ_TIMEOUT_S
    assert EMBED_CONNECT_TIMEOUT_S == 5.0
    assert EMBED_READ_TIMEOUT_S == 20.0


def test_multi_facet_deadline_constant():
    from agent import MULTI_FACET_DEADLINE_S

    assert 10.0 <= MULTI_FACET_DEADLINE_S <= 30.0


def test_kb_disk_cache_memoizes():
    from kb.store import load_kb, load_embeddings, clear_kb_disk_cache, _load_kb_cached

    clear_kb_disk_cache()
    kb1 = load_kb()
    kb2 = load_kb()
    assert kb1 is kb2
    info = _load_kb_cached.cache_info()
    assert info.hits >= 1
    emb1 = load_embeddings()
    emb2 = load_embeddings()
    assert emb1 is not None and emb2 is not None
    # same matrix object from cache
    assert emb1[0] is emb2[0]
    clear_kb_disk_cache()
    kb3 = load_kb()
    assert kb3 is not None
    # after clear, new load is a different object
    assert kb3 is not kb1

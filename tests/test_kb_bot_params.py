"""Unit tests for Phase 3 structured bot params in KB extraction."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import BOTS_DIR
from kb.extract import build_knowledge_base, extract_bots
from kb.schema import KnowledgeBase


@pytest.mark.skipif(not BOTS_DIR or not os.path.isdir(BOTS_DIR), reason="BOTS_DIR missing")
def test_extract_bots_includes_kafka_read_stream_params():
    kb = KnowledgeBase()
    extract_bots(kb)
    bots = [e for e in kb.entities if e.kind == "bot" and "read-stream" in e.name.lower()]
    kafka = [b for b in bots if "kafka-v2" in b.name.lower()]
    assert kafka, "expected @kafka-v2:read-stream entity"
    bot = kafka[0]
    params = (bot.metadata or {}).get("parameters") or []
    names = {p.get("name") for p in params}
    assert {"name", "group", "offset_reset", "batch_size"} <= names
    facts = [f for f in kb.facts if f.entity_id == bot.id and "parameters" in f.text.lower()]
    assert facts, "expected fact-params entry for kafka-v2:read-stream"


@pytest.mark.skipif(not BOTS_DIR or not os.path.isdir(BOTS_DIR), reason="BOTS_DIR missing")
def test_searchable_entries_include_param_names():
    kb = KnowledgeBase()
    extract_bots(kb)
    entries = kb.searchable_entries()
    kafka_entries = [
        e for e in entries
        if e.get("kind") == "bot" and "kafka-v2" in (e.get("title") or "").lower()
        and "read-stream" in (e.get("title") or "").lower()
    ]
    assert kafka_entries
    text = kafka_entries[0]["text"].lower()
    assert "batch_size" in text or "batch" in text
    assert "parameters" in text or "offset_reset" in text


@pytest.mark.skipif(not BOTS_DIR or not os.path.isdir(BOTS_DIR), reason="BOTS_DIR missing")
def test_timed_loop_params_extracted():
    kb = KnowledgeBase()
    extract_bots(kb)
    loops = [
        e for e in kb.entities
        if e.kind == "bot" and "timed-loop" in e.name.lower()
    ]
    assert loops
    names = set((loops[0].metadata or {}).get("param_names") or [])
    assert "interval" in names or any(
        "interval" in (p.get("name") or "")
        for p in ((loops[0].metadata or {}).get("parameters") or [])
    )

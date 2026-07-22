"""Unit tests for extension-family bot parameter fast path."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bot_lookup import (
    bot_family_hints,
    bot_operation_hints,
    format_param_answer,
    is_bot_param_lookup,
    lookup_bot_params_from_catalog,
    pick_bot_section,
    split_bot_catalog_sections,
)
from query_qdrant import parse_bot_param_table

KAFKA_SNIPPET = """
## Bot @kafka-v2:alter-topic

| Parameter Name | Type | Default Value | Description |
| --- | --- | --- | --- |
| name* | Text |  | Kafka topic name |

---

## Bot @kafka-v2:read-stream

| Parameter Name | Type | Default Value | Description |
| --- | --- | --- | --- |
| name* | Text |  | Stream name |
| group* | Text |  | Data receiver (consumer) group name |
| offset_reset | Text | earliest | Stream offset reset position |
| batch_size | Text | 100 | Maximum rows to read in each batch |
"""


def test_is_bot_param_lookup():
    assert is_bot_param_lookup("What parameters does the timed-loop bot take?")
    assert is_bot_param_lookup("What does @kafka-v2:read-stream accept?")
    assert not is_bot_param_lookup("Walk me through wiring Kafka into Fabrix")


def test_bot_family_hints_kafka_v2_variants():
    hints = bot_family_hints("What parameters does the kafka v2 consume bot take?")
    assert "kafka-v2" in hints or "kafka_v2" in hints


def test_bot_operation_hints_read_consume():
    ops = bot_operation_hints("kafka-v2 consume / read bot parameters")
    assert "read-stream" in ops


def test_pick_bot_section_prefers_read_stream():
    sections = split_bot_catalog_sections(KAFKA_SNIPPET)
    picked = pick_bot_section(sections, ["kafka-v2"], ["read-stream"])
    assert picked is not None
    assert "read-stream" in picked["bot_name"].lower()


def test_parse_bot_param_table_required_marker():
    table = """| Parameter Name | Type | Default Value | Description |
| --- | --- | --- | --- |
| name\\* | Text |  | Stream name |
| group\\* | Text |  | Group name |"""
    rows = parse_bot_param_table(table)
    assert len(rows) >= 2
    assert rows[0]["name"] == "name"
    assert rows[0]["required"] is True


def test_format_param_answer_includes_bot_name():
    rows = [{"name": "interval", "required": True, "type": "Text", "default": "", "description": "Seconds"}]
    ans = format_param_answer("@c:timed-loop", rows)
    assert "timed-loop" in ans
    assert "interval" in ans


@pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "config.py",
        )
    ),
    reason="config missing",
)
def test_lookup_bot_params_from_catalog_kafka_v2():
    from config import BOTS_DIR

    if not os.path.isfile(os.path.join(BOTS_DIR, "kafka-v2.md")):
        pytest.skip("BOTS_DIR kafka-v2.md missing")
    result = lookup_bot_params_from_catalog("kafka-v2", ["read-stream"])
    assert result is not None
    bot_name, rows, rel_path = result
    assert "read-stream" in bot_name.lower()
    names = {r["name"] for r in rows}
    assert {"name", "group", "offset_reset", "batch_size"} <= names

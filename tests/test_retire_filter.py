#!/usr/bin/env python3
"""Retired-source filter on chunk/KB retrieve shapes."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import freshness as fr  # noqa: E402


class RetireRetrieveShapeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "retired.json")
        fr.RETIRED_PATH = self.path

    def tearDown(self):
        self.tmp.cleanup()
        fr.RETIRED_PATH = fr.RETIRED_SOURCES_PATH

    def test_filter_kb_entries_by_source(self):
        fr.retire_source("Bots/cfxdm.md", path=self.path)
        entries = [
            {"id": "1", "source": "Bots/cfxdm.md", "text": "secret"},
            {"id": "2", "source": "Bots/other.md", "text": "ok"},
        ]
        out = fr.filter_retired_entries(entries, fr.retired_set(self.path))
        self.assertEqual([e["id"] for e in out], ["2"])

    def test_unretire_restores(self):
        fr.retire_source("Bots/secret.md", path=self.path)
        fr.unretire_source("Bots/secret.md", path=self.path)
        chunks = [{"text": "x", "metadata": {"source": "Bots/secret.md"}}]
        self.assertEqual(len(fr.filter_retired_chunks(chunks, fr.retired_set(self.path))), 1)


if __name__ == "__main__":
    unittest.main()

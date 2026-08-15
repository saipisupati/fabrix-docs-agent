#!/usr/bin/env python3
"""Unit tests for scrape freshness diffs, retired-source filtering, and lock flag."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import freshness as fr  # noqa: E402


class HashAndDiffTests(unittest.TestCase):
    def test_hash_bytes_stable(self):
        self.assertEqual(fr.hash_bytes(b"hello"), fr.hash_bytes(b"hello"))
        self.assertNotEqual(fr.hash_bytes(b"hello"), fr.hash_bytes(b"Hello"))

    def test_diff_detects_changed_added_removed(self):
        old = {
            "pages_meta": {
                "a.md": {"path": "a.md", "sha256": "111", "bytes": 1, "scraped_at": "t0", "status": "ok"},
                "gone.md": {"path": "gone.md", "sha256": "222", "bytes": 1, "scraped_at": "t0", "status": "ok"},
            }
        }
        new = {
            "pages_meta": {
                "a.md": {"path": "a.md", "sha256": "999", "bytes": 2, "scraped_at": "t1", "status": "ok"},
                "b.md": {"path": "b.md", "sha256": "333", "bytes": 1, "scraped_at": "t1", "status": "ok"},
            }
        }
        d = fr.diff_manifests(old, new)
        self.assertEqual(d["changed_paths"], ["a.md"])
        self.assertEqual(d["added_paths"], ["b.md"])
        self.assertEqual(d["removed_paths"], ["gone.md"])
        self.assertTrue(d["has_content_changes"])

    def test_should_rebuild(self):
        self.assertTrue(fr.should_rebuild({"has_content_changes": True}))
        self.assertFalse(fr.should_rebuild({"has_content_changes": False, "fail": 0}))
        self.assertTrue(fr.should_rebuild(None))


class RetireFilterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "retired.json")
        fr.RETIRED_PATH = self.path

    def tearDown(self):
        self.tmp.cleanup()
        fr.RETIRED_PATH = fr.RETIRED_SOURCES_PATH

    def test_filter_retired_chunks(self):
        fr.retire_source("Bots/secret.md", path=self.path)
        chunks = [
            {"text": "a", "metadata": {"source": "Bots/secret.md"}},
            {"text": "b", "metadata": {"source": "Bots/cfxdm.md"}},
        ]
        out = fr.filter_retired_chunks(chunks, fr.retired_set(self.path))
        self.assertEqual([c["metadata"]["source"] for c in out], ["Bots/cfxdm.md"])
        fr.unretire_source("Bots/secret.md", path=self.path)
        out2 = fr.filter_retired_chunks(chunks, fr.retired_set(self.path))
        self.assertEqual(len(out2), 2)

    def test_basename_retire(self):
        fr.retire_source("cfxdm.md", path=self.path)
        chunks = [{"text": "x", "metadata": {"source": "Bots/cfxdm.md"}}]
        out = fr.filter_retired_chunks(chunks, fr.retired_set(self.path))
        self.assertEqual(out, [])


class StatusLockTests(unittest.TestCase):
    def test_qdrant_lock_held(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(fr.qdrant_lock_held(d))
            open(os.path.join(d, ".lock"), "w").close()
            self.assertTrue(fr.qdrant_lock_held(d))

    def test_load_kb_status_includes_qdrant_locked(self):
        status = fr.load_kb_status(qdrant_client=None)
        self.assertIn("qdrant_locked", status)
        self.assertIsInstance(status["qdrant_locked"], bool)
        self.assertIn("retired", status)


if __name__ == "__main__":
    unittest.main()

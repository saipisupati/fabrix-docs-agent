#!/usr/bin/env python3
"""Unit tests for scrape freshness diffs and retired-source filtering."""

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

    def test_diff_unchanged(self):
        meta = {"a": {"sha256": "aaa"}}
        d = fr.diff_manifests({"pages_meta": meta}, {"pages_meta": dict(meta)})
        self.assertFalse(d["has_content_changes"])

    def test_should_rebuild(self):
        self.assertTrue(fr.should_rebuild({"has_content_changes": True}))
        self.assertFalse(fr.should_rebuild({"has_content_changes": False, "fail": 0}))
        self.assertTrue(fr.should_rebuild({"ok": 10, "page_paths": ["a"]}))
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

    def test_empty_retired_noop(self):
        chunks = [{"text": "x", "metadata": {"source": "Bots/cfxdm.md"}}]
        self.assertEqual(fr.filter_retired_chunks(chunks, set()), chunks)


if __name__ == "__main__":
    unittest.main()

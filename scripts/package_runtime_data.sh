#!/usr/bin/env bash
# Package local Qdrant + KB artifacts for CI (upload as a release asset or host URL).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d data/qdrant_db ]]; then
  echo "missing data/qdrant_db — run ingest first" >&2
  exit 1
fi
if [[ ! -f data/kb/kb.json ]]; then
  echo "missing data/kb/kb.json — run build_kb.py first" >&2
  exit 1
fi
tar czf fabrix_runtime_data.tar.gz data/qdrant_db data/kb
echo "Wrote fabrix_runtime_data.tar.gz ($(du -h fabrix_runtime_data.tar.gz | cut -f1))"

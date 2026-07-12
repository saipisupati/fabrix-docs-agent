"""
ingest_and_test_remote.py, alternate path: upload md to a hosted Qdrant endpoint.

Server does chunking + BGE-large embedding. Not the primary local path.
Use ingest_qdrant.py for day-to-day work unless you need shared server storage.
"""

import os
import sys
import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import BOTS_DIR, CFXQL_FILE, COLLECTION_NAME, REMOTE_BASE_URL

EMBEDDING_MODEL_REMOTE = os.environ.get(
    "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
)

SAMPLE_QUESTIONS = [
    "what does the kafka poll-topic bot do?",
    "what is the difference between full and restricted cfxql?",
    "what parameters does the jira create-issue bot take?",
]


def _base_url():
    url = (REMOTE_BASE_URL or "").rstrip("/")
    if not url:
        print("Error: REMOTE_BASE_URL is required (set in .env).")
        sys.exit(1)
    return url


def create_collection():
    base = _base_url()
    print(f"Creating collection '{COLLECTION_NAME}' with model '{EMBEDDING_MODEL_REMOTE}'...")
    response = requests.post(
        f"{base}/collections",
        headers={"Content-Type": "application/json"},
        json={"collection_name": COLLECTION_NAME, "embedding_model": EMBEDDING_MODEL_REMOTE},
    )
    print(f"  -> {response.status_code}: {response.text[:200]}")
    return response.ok


def upload_file(filepath, timeout=120):
    filename = os.path.basename(filepath)
    base = _base_url()
    with open(filepath, "rb") as f:
        try:
            response = requests.post(
                f"{base}/upload",
                data={"collection_name": COLLECTION_NAME},
                files={"file": (filename, f)},
                timeout=timeout,
            )
            return response.ok, response.status_code
        except requests.exceptions.Timeout:
            return False, "TIMEOUT"
        except requests.exceptions.RequestException as e:
            return False, f"ERROR: {e}"

def ingest_all():
    print("\nIngesting CFXQL reference...")
    ok, status = upload_file(CFXQL_FILE)
    print(f"  -> {status} {'OK' if ok else 'FAILED'}")

    print(f"\nIngesting bot catalog from BOTS_DIR={BOTS_DIR} ...")
    md_files = sorted(f for f in os.listdir(BOTS_DIR) if f.endswith(".md"))
    succeeded, failed = 0, 0
    for i, filename in enumerate(md_files, 1):
        filepath = os.path.join(BOTS_DIR, filename)
        ok, status = upload_file(filepath)
        if ok:
            succeeded += 1
        else:
            failed += 1
            print(f"  FAILED on {filename}: status {status}")
        if i % 25 == 0 or i == len(md_files):
            print(f"  [{i}/{len(md_files)}] uploaded so far: {succeeded} ok, {failed} failed")

    print(f"\nIngestion done: {succeeded} succeeded, {failed} failed out of {len(md_files)} bot files")


def search(question, limit=3):
    base = _base_url()
    response = requests.post(
        f"{base}/search",
        headers={"Content-Type": "application/json"},
        json={"collection_name": COLLECTION_NAME, "query": question, "limit": limit},
    )
    if not response.ok:
        print(f"  Search failed: {response.status_code} {response.text[:200]}")
        return []
    data = response.json()
    return data.get("results", [])


def run_sample_questions():
    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)
    for question in SAMPLE_QUESTIONS:
        print(f'\nQuestion: "{question}"')
        results = search(question)
        for i, r in enumerate(results, 1):
            score = r.get("score", "?")
            meta = r.get("metadata", {})
            source = meta.get("source_file", "?")
            heading = meta.get("heading", "?")
            text_preview = r.get("text", "")[:80]
            print(f"  {i}. score={score:.3f}  source={source}  heading={heading}")
            print(f"     text: {text_preview}...")


def main():
    print(f"Using REMOTE_BASE_URL={_base_url()}")
    print(f"Using BOTS_DIR={BOTS_DIR}")
    print(f"Using CFXQL_FILE={CFXQL_FILE}")
    if not create_collection():
        print("Collection creation failed, stopping.")
        sys.exit(1)

    ingest_all()
    run_sample_questions()


if __name__ == "__main__":
    main()

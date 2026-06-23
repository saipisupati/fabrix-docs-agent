"""

test_bge_large_local.py : focused local test of BAAI/bge-large-en-v1.5 across several real questions, since the remote Qdrant server isn't reachable yet without VPN access.

Requires: OPENROUTER_API_KEY environment variable set.

"""

import os

import sys

import requests

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from langchain_text_splitters import MarkdownHeaderTextSplitter

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

MODEL = "baai/bge-large-en-v1.5"

TEST_QUESTIONS = [

    {

        "question": "what is the difference between full and restricted cfxql?",

        "expect_headers": {"Full CFXQL", "Restricted CFXQL"},

    },

    {

        "question": "what operators does restricted cfxql support?",

        "expect_headers": {"Restricted CFXQL"},

    },

    {

        "question": "how do i check if a value is null in cfxql?",

        "expect_headers": {"Unary Operators for NULL Value Checks"},

    },

]

def chunk_markdown_file(filepath):

    with open(filepath, encoding="utf-8") as f:

        text = f.read()

    headers_to_split_on = [("##", "h2"), ("####", "h4")]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    return splitter.split_text(text)

def embed_texts(texts):

    response = requests.post(

        EMBEDDINGS_URL,

        headers={

            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",

            "Content-Type": "application/json",

        },

        json={"model": MODEL, "input": texts},

    )

    if response.status_code != 200:

        raise RuntimeError(f"Embedding request failed ({response.status_code}): {response.text}")

    data = response.json()["data"]

    data.sort(key=lambda d: d["index"])

    return [d["embedding"] for d in data]

def main():

    if len(sys.argv) < 2:

        print("Usage: python3 test_bge_large_local.py /path/to/cfxql.md")

        sys.exit(1)

    filepath = sys.argv[1]

    chunks = chunk_markdown_file(filepath)

    print(f"Loaded {len(chunks)} chunks from {filepath}")

    texts = [c.page_content for c in chunks]

    print(f"Embedding all chunks with {MODEL}...")

    chunk_vectors = embed_texts(texts)

    for test in TEST_QUESTIONS:

        question = test["question"]

        expected = test["expect_headers"]

        print(f"\n{'='*70}")

        print(f'Question: "{question}"')

        print(f"Expected to rank highly: {expected}")

        print('='*70)

        query_vector = embed_texts([question])[0]

        sims = cosine_similarity([query_vector], chunk_vectors)[0]

        ranked = sorted(enumerate(sims), key=lambda x: -x[1])

        for rank, (idx, score) in enumerate(ranked[:5], 1):

            h2 = chunks[idx].metadata.get("h2", "none")

            h4 = chunks[idx].metadata.get("h4", "")

            label = h4 if h4 else h2

            hit = " <-- EXPECTED" if label in expected else ""

            print(f"  {rank}. score={score:.3f}  {label}{hit}")

if __name__ == "__main__":

    main()

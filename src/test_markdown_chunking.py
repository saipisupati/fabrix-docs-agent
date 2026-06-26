"""
test_markdown_chunking.py, DEV ONLY: try MarkdownHeaderTextSplitter on cfxql.md.

Chunking experiment: python3 src/test_markdown_chunking.py /path/to/cfxql.md
"""

import sys
from langchain_text_splitters import MarkdownHeaderTextSplitter


def chunk_markdown_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    headers_to_split_on = [
        ("##", "h2"),
        ("####", "h4"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(text)
    return chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_markdown_chunking.py /path/to/file.md")
        sys.exit(1)

    filepath = sys.argv[1]
    chunks = chunk_markdown_file(filepath)

    print(f"\nFile: {filepath}")
    print(f"Produced {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks):
        print(f"--- chunk {i} ---")
        print(f"  metadata: {chunk.metadata}")
        preview = chunk.page_content[:100].replace("\n", " ")
        print(f"  preview: {preview}...")
        print(f"  length: {len(chunk.page_content)} chars")
        print()


if __name__ == "__main__":
    main()

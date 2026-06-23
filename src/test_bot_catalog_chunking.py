"""
test_bot_catalog_chunking.py: cleans a bot catalog markdown file, then
splits it into one chunk per bot using MarkdownHeaderTextSplitter,
splitting on ## (the level each individual bot is marked at).
"""

import sys
from langchain_text_splitters import MarkdownHeaderTextSplitter
from clean_markdown import clean_markdown, extract_bot_metadata


def chunk_bot_catalog_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_markdown(raw_text)
    
    headers_to_split_on = [("##", "h2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(cleaned_text)

    # enrich each chunk's metadata with clean bot_name and prefix
    for chunk in chunks:
        h2 = chunk.metadata.get("h2", "")
        if h2.startswith("Bot "):
            bot_meta = extract_bot_metadata(h2)
            chunk.metadata.update(bot_meta)

    return chunks

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_bot_catalog_chunking.py /path/to/file.md")
        sys.exit(1)

    filepath = sys.argv[1]
    chunks = chunk_bot_catalog_file(filepath)

    print(f"Produced {len(chunks)} chunks\n")
    print("First 5 chunks:\n")
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- chunk {i} ---")
        print(f"  metadata: {chunk.metadata}")
        preview = chunk.page_content[:120].replace("\n", " ")
        print(f"  preview: {preview}...")
        print(f"  length: {len(chunk.page_content)} chars")
        print()


if __name__ == "__main__":
    main()


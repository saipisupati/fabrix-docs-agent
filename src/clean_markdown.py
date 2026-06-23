"""
clean_markdown.py: strips unnecessary text from raw markdown files before chunking:
YAML frontmatter, <style> blocks, and inline HTML tags (keeping their
text content). This is needed because bot catalog pages contain CSS
styling and HTML spans that would otherwise take up chunk text.
"""

import re


def clean_markdown(text):
    """Remove frontmatter, style blocks, and HTML tags from raw markdown."""
    # Step 1: remove YAML frontmatter (--- ... --- at the very top)
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)

    # Step 2: remove <style>...</style> blocks entirely
    text = re.sub(r'<style>.*?</style>', '', text, flags=re.DOTALL)

    # Step 3: strip HTML tags but keep their text content
    text = re.sub(r'<[^>]+>', '', text)

    return text

def extract_bot_metadata(h2_header):
    """Given a header like 'Bot @dm:add-bounded-dataset', extract just
    the bot name and its prefix character (@, #, or *)."""
    if not h2_header.startswith("Bot "):
        return {"bot_name": "unknown", "prefix": "unknown"}

    bot_name = h2_header[len("Bot "):].strip()
    prefix = bot_name[0] if bot_name and bot_name[0] in "@#*" else "unknown"

    return {"bot_name": bot_name, "prefix": prefix}



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 clean_markdown.py /path/to/file.md")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_markdown(raw)
    print(f"Original length: {len(raw)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
    print(f"\n--- First 500 chars of cleaned output ---\n")
    print(cleaned[:500])

"""
chunk_heuristic.py — a generalizable fallback chunker for plain text docs
that don't have markdown structure available.

HONEST LIMITATIONS (read this before relying on it):
Header detection from plain text alone is inherently fuzzy. Testing this
against the real CFXQL reference doc surfaced two failure modes:
  1. FALSE POSITIVES: short lines that aren't headers but look like one
     (e.g. "port gt 80", "device is empty" - these are example query
     lines, not section titles)
  2. FALSE NEGATIVES: real headers that get missed because the heuristic's
     thresholds don't fit every case

This means: USE THIS as a fallback when no markdown source is available,
NOT as a replacement for proper structure-aware chunking. If real
markdown becomes available, MarkdownHeaderTextSplitter (LangChain) will
be far more reliable, since it reads actual header markup (#, ##) instead
of guessing from formatting patterns.
"""

import re


def is_likely_header(lines, i, max_header_words=8, min_body_words=6):
    """Heuristic: a line is a likely header if it's short, doesn't end in
    sentence punctuation, doesn't look like code/query syntax, AND is
    followed within a couple lines by a longer sentence that elaborates
    on it."""
    line = lines[i].strip()
    if not line:
        return False

    word_count = len(line.split())
    ends_with_punct = line and line[-1] in '.!?:'
    looks_like_code = bool(re.search(r"[='\"\[\]{}()<>]", line))

    if word_count > max_header_words or ends_with_punct or looks_like_code:
        return False

    for j in range(i + 1, min(i + 3, len(lines))):
        next_line = lines[j].strip()
        if not next_line:
            continue
        next_word_count = len(next_line.split())
        return next_word_count >= min_body_words
    return False


def chunk_by_heuristic_sections(text, source_name, max_chunk_chars=1200):
    """Split text into sections using the header heuristic, then further
    split any section that's still too long using a simple size-based
    fallback."""
    lines = text.split("\n")
    header_indices = [i for i in range(len(lines)) if is_likely_header(lines, i)]

    if not header_indices:
        return [{
            "text": text.strip(),
            "metadata": {"source": source_name, "type": "narrative",
                         "section_header": "n/a", "chunking_method": "heuristic_no_headers_found"}
        }]

    chunks = []
    for idx, header_line_num in enumerate(header_indices):
        start = header_line_num
        end = header_indices[idx + 1] if idx + 1 < len(header_indices) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()
        if not section_text:
            continue

        header_text = lines[header_line_num].strip()

        if len(section_text) > max_chunk_chars:
            for sub_start in range(0, len(section_text), max_chunk_chars):
                sub_text = section_text[sub_start:sub_start + max_chunk_chars]
                chunks.append({
                    "text": sub_text.strip(),
                    "metadata": {"source": source_name, "type": "narrative",
                                 "section_header": header_text,
                                 "chunking_method": "heuristic"}
                })
        else:
            chunks.append({
                "text": section_text,
                "metadata": {"source": source_name, "type": "narrative",
                             "section_header": header_text,
                             "chunking_method": "heuristic"}
            })

    return chunks


if __name__ == "__main__":
    import os
    DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    with open(os.path.join(DOCS_DIR, "cfxql_reference.txt")) as f:
        text = f.read()

    chunks = chunk_by_heuristic_sections(text, "cfxql_reference.txt")
    print(f"Heuristic chunking produced {len(chunks)} chunks\n")
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        preview = c["text"][:80].replace("\n", " ")
        print(f"[{i}] header=\"{meta['section_header']}\" ({len(c['text'])} chars)")
        print(f"     {preview}...")

"""
chunk_heuristic.py, fallback chunker when you only have plain text, no markdown headers.

Guesses section breaks from line length/punctuation. Noisier than real markdown splitting;
use MarkdownHeaderTextSplitter when .md source is available (see ingest_qdrant.py).
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
    # find likely headers, split sections, size-split anything still too long
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

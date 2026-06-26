"""
query.py, LEGACY v0: query Chroma + TF-IDF store from ingest.py.

Use query_qdrant.py / agent.py instead. Kept for reference only.
"""

import os
import sys
import pickle
import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
VECTORIZER_PATH = os.path.join(CHROMA_DIR, "vectorizer.pkl")
COLLECTION_NAME = "fabrix_docs"


def load_vectorizer():
    with open(VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def retrieve(question, vectorizer, collection, top_k=3, filter=None):
    # ============================================================
    # SWAP POINT: replace with the real embedding model, e.g.:
    #   from langchain_mistralai import MistralAIEmbeddings
    #   embedder = MistralAIEmbeddings(model="mistral-embed")
    #   query_vector = [embedder.embed_query(question)]
    # ============================================================
    query_vector = vectorizer.transform([question]).toarray().tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        where=filter,
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "metadata": meta})
    return chunks


def build_prompt(question, retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source")
        context_parts.append(f"[{i}] Source: {source}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant for Fabrix.ai documentation.
Answer the user's question using ONLY the documentation excerpts provided below.
If the answer is not in the excerpts, say "I couldn't find that in the documentation."
Always cite which excerpt your answer comes from using [1], [2], etc.

DOCUMENTATION EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:"""


def generate(prompt):
    # ============================================================
    # SWAP POINT: replace with the real model call once confirmed, e.g.:
    #
    #   from langchain_mistralai import ChatMistralAI
    #   llm = ChatMistralAI(model="mistral-large-latest")
    #   return llm.invoke(prompt).content
    #
    # ============================================================
    return "[no LLM configured yet - see generate() in query.py]"


def ask(question, top_k=3, filter=None):
    vectorizer = load_vectorizer()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    chunks = retrieve(question, vectorizer, collection, top_k=top_k, filter=filter)

    print(f"Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        label = meta.get("bot_name") if meta.get("type") == "bot" else meta.get("source")
        print(f"  [{i}] {label} (type={meta['type']}, cfxql={meta['cfxql_type']})")

    prompt = build_prompt(question, chunks)
    answer = generate(prompt)

    print(f"\n--- ANSWER ---\n{answer}")
    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/query.py "your question here"')
        sys.exit(1)
    question = sys.argv[1]
    ask(question)

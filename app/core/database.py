from typing import Any, List
from uuid import uuid4
import ollama
import chromadb
from app.core.config import settings


def _extract_embedding(response: Any) -> list[float]:
    if not response:
        raise RuntimeError(
            "Ollama returned no response. Ensure Ollama is running and the embedding model is available."
        )

    if isinstance(response, dict):
        embedding = response.get("embedding")
        keys_info = list(response.keys())
    else:
        embedding = getattr(response, "embedding", None)
        keys_info = [name for name in dir(response) if not name.startswith("_")]

    if embedding is None:
        raise RuntimeError(
            f"Ollama response did not include 'embedding'. Response fields: {keys_info}"
        )

    return embedding


def create_embeddings(chunk_dict: List[dict]) -> List[dict]:
    """Converts text chunks into vectors using Ollama."""
    process_data = []

    for file_entry in chunk_dict:
        file_name = file_entry.get("filename")
        chunk_list = file_entry.get("chunk", [])

        print(f"🧠 Generating embeddings for {len(chunk_list)} chunks from {file_name}")

        for idx, text_content in enumerate(chunk_list):
            # API Call to Ollama
            text_content = text_content.strip()

            if not text_content:
                print(f"⏩ Skipping empty chunk {idx}")

            try:

                response = ollama.embeddings(
                    model=settings.EMBED_MODEL,
                    prompt=text_content,
                    options={"num_ctx": 2048},
                )
            except Exception as e:
                print(f"⚠️ Error embedding chunk in {file_name}: {e}")
                continue

            embedding = _extract_embedding(response)

            process_data.append(
                {
                    "id": f"{file_name}_{uuid4()}",
                    "vector": embedding,
                    "document": text_content,
                    "metadata": {"source": file_name, "chunk_index": idx},
                }
            )

    return process_data


def save_to_chroma(processed_data: List[dict]) -> str:
    """Persists the embedded data into ChromaDB."""
    if not processed_data:
        return "⚠️ No data provided to save."

    # Initialize Client
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

    try:
        client.delete_collection(name="pdf_knowledge_base")
    except:
        pass

    collection = client.get_or_create_collection(name="pdf_knowledge_base")

    # Unzip the dictionaries into lists for ChromaDB
    ids = [item["id"] for item in processed_data]
    vectors = [item["vector"] for item in processed_data]
    metadata = [item["metadata"] for item in processed_data]
    documents = [item["document"] for item in processed_data]

    # Batch upload
    collection.upsert(
        ids=ids, embeddings=vectors, metadatas=metadata, documents=documents
    )

    # Print some useful debugging info
    print(f"✅ Succeeded! Processed {len(ids)} vector entries.")
    print(f"First ID: {ids[0]}")
    print(f"Vector length: {len(vectors[0])}")

    return f"💾 Successfully saved to {settings.CHROMA_PATH}"


# Add a separate Query function here for later use
def query_rag(query_text: str, n_results: int = 3):
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    collection = client.get_collection(name="pdf_knowledge_base")

    # Embed the question
    response = ollama.embeddings(model=settings.EMBED_MODEL, prompt=query_text)
    query_vec = _extract_embedding(response)

    return collection.query(query_embeddings=[query_vec], n_results=n_results)

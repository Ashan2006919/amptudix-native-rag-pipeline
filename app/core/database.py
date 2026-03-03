from typing import Any, List, Optional
import ollama
import chromadb
from app.core.config import settings
from app.core.file_engine import recursive_split


async def _extract_embedding(response: Any) -> list[float]:
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


async def create_embeddings(chunk_dict: List[dict]) -> List[dict]:
    """Converts text chunks into vectors using Ollama."""
    processed_data = []

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

            embedding = await _extract_embedding(response)

            processed_data.append(
                {
                    "id": f"{file_name}_{idx}",
                    "vector": embedding,
                    "document": text_content,
                    "metadata": {"source": file_name, "chunk_index": idx},
                }
            )

    return processed_data


async def save_to_chroma(
    processed_data: List[dict], filename: Optional[str] = None
) -> str:
    """Persists the embedded data into ChromaDB."""
    if not processed_data:
        return "⚠️ No data provided to save."

    # Initialize Client
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    collection = client.get_or_create_collection(name="pdf_knowledge_base")
    existing_count = collection.count()

    if existing_count > 0:
        sample = collection.get(limit=1, include=["embeddings"])
        if sample["embeddings"]:
            current_dim = len(sample["embeddings"])
            print(f"Database dimension: {current_dim}")

    if filename:
        print(f"🧹 Clean old entries for: {filename}")
        collection.delete(where={"source": filename})
    else:
        print(f"🧹 Deleting everything before inserting...")
        client.delete_collection(name="pdf_knowledge_base")

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


async def process_one_file(filename: str):
    chunk_dict = await recursive_split(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        filename=filename,
    )

    processed_data = await create_embeddings(chunk_dict)

    await save_to_chroma(processed_data=processed_data, filename=filename)

    print(f"🎉 Background indexing complete for {filename}")


# Add a separate Query function here for later use
async def query_rag(query_text: str, n_results: int = 3):
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    collection = client.get_collection(name="pdf_knowledge_base")

    # Embed the question
    response = await ollama.AsyncClient().embeddings(
        model=settings.EMBED_MODEL, prompt=query_text
    )
    query_vec = await _extract_embedding(response)

    return collection.query(query_embeddings=[query_vec], n_results=n_results)

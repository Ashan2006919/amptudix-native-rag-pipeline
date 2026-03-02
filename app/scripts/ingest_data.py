from typing import List
from uuid import uuid4
import ollama
import chromadb
from app.core.config import settings
from app.core.pdf_engine import recursive_split


def create_embeddings(chunk_dict: List[dict]) -> List[dict]:
    process_data = []

    for file_entry in chunk_dict:
        file_name = file_entry.get("filename")
        chunk_list = file_entry.get("chunk", [])

        print(f"🧠 Genarating embedding for {len(chunk_list)} chunk from {file_name}")

        for idx, text_content in enumerate(chunk_list):
            response = ollama.embeddings(
                model=settings.EMBED_MODEL, prompt=text_content
            )

            process_data.append(
                {
                    "id": f"{file_name}_{uuid4()}",
                    "vector": response["embedding"],
                    "document": text_content,
                    "metadata": {"source": file_name, "chunk_index": idx},
                }
            )

    return process_data


processed_embeddings = create_embeddings(
    recursive_split(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
)


def save_to_chroma(processed_data: List[dict]) -> str:
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

    collection = client.get_or_create_collection(name="pdf_knowledge_base")

    ids = [item["id"] for item in processed_data]
    vectors = [item["vector"] for item in processed_data]
    metadata = [item["metadata"] for item in processed_data]
    documents = [item["document"] for item in processed_data]

    collection.upsert(
        ids=ids, embeddings=vectors, metadatas=metadata, documents=documents
    )

    return f"💾 Successfully saved {len(ids)} to {settings.CHROMA_PATH}"


if processed_embeddings:
    print(f"✅ Succeeded! Created {len(processed_embeddings)} total vector entries!\n")
    print(f"⏱️ Saving vectors to the chroma database.....")
    print(save_to_chroma(processed_embeddings))
    print(f"First ID: {processed_embeddings[0]["id"]}")
    print(f"Vector length: {len(processed_embeddings[0]["vector"])}")

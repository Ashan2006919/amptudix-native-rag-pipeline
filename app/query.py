from chromadb import PersistentClient
import ollama
from app.core.config import settings


def query_database(query_text: str, n_results: int = 3):
    client = PersistentClient(path=settings.CHROMA_PATH)

    collection = client.get_collection("pdf_knowledge_base")

    query_embedding = ollama.embeddings(model=settings.EMBED_MODEL, prompt=query_text)[
        "embedding"
    ]

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    return results

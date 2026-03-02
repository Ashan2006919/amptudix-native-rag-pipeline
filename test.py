from ollama import embeddings
from app.core.database import _extract_embedding

response = embeddings(
    model="mxbai-embed-large:latest",
    prompt="""This document is an extended synthetic knowledge base built for testing Retrieval Augmented
Generation (RAG) pipelines. It contains detailed structured and unstructured information about
Ashan Niwantha including personal growth, academic goals, technical projects, creative ambitions,
philosophical reflections, habits, and long-term vision. The purpose is to create sufficient density
and variation for chunking, embedding, and retrieval evaluation.
""",
)

embedding = _extract_embedding(response)

print(embeddings)

import asyncio
from app.core.config import settings
from app.core.file_engine import recursive_split
from app.core.database import create_embeddings, save_to_chroma, query_rag
from app.core.chat import generate_answer


async def main():
    # # Use settings instead of hardcoded numbers
    # chunk_dict = await recursive_split(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    # # Process and Save
    # processed_data = await create_embeddings(chunk_dict)
    # await save_to_chroma(processed_data)

    # Simple Query Interface
    while True:
        query = input("\n🔍 Query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        response = await query_rag(query)

        # ChromaDB returns a nested dict. Let's make the print readable:
        print("\n📚 Results:")
        documents = (response or {}).get("documents") or []
        for doc in documents[0] if documents else []:
            print(f"- {doc[:200]}...")


async def test():
    while True:
        query = input("\nYou: ")

        if query.lower() == exit:
            break

        await generate_answer(query)


if __name__ == "__main__":
    asyncio.run(main())

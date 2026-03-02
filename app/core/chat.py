import ollama
from app.core.config import settings
from app.core.database import query_rag


async def generate_answer(query: str):
    search_results = query_rag(query, 6)
    documents = (search_results or {}).get("documents") or []
    context_chunks = documents[0] if documents else []
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You're an helpful AI asistant that has a sense of humer. Use the following retrived context for an questions as by the user.
    Use those two to give a meaningful answer. If you can't provide an answer based on given context then say I don't know.
    Don't try to guess the answer.
    
    quetion: {query}
    
    Context:
    {context}
    
    ANSWER:
    """

    response = ollama.generate(model="llama3.2:3b", prompt=prompt, stream=True)

    print("\n🚀 Llama is thinking....")
    for chunk in response:
        print(chunk["response"], end="", flush=True)

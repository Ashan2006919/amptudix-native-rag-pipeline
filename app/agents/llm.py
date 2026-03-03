from typing import List
import ollama
from google import genai
from app.core.config import settings


async def get_answers(prompt: str, query: str, history: dict):

    messages = [{"role": "system", "content": prompt}]
    messages.extend(history)

    messages.append({"role": "user", "content": query})

    async for chunk in ollama_model(messages=messages):
        yield chunk


async def ollama_model(messages: List):
    client = ollama.AsyncClient()

    async for part in await client.chat(
        model=settings.OLLAMA_MODEL_NAME, stream=True, messages=messages
    ):
        yield part["message"]["content"]


async def google_model(prompt: str, query: str) -> dict:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{prompt}\n\nUser Question: {query}",
    )

    answer = (getattr(response, "text", None) or "").strip()

    if not answer:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            answer = " ".join(
                getattr(part, "text", "")
                for part in parts
                if getattr(part, "text", None)
            ).strip()

    return {"answer": answer or "I don't know"}

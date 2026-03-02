import ollama
from google import genai
from app.core.config import settings


async def get_answers(prompt: str, query: str) -> dict:
    return await ollama_model(prompt=prompt, query=query)


async def ollama_model(prompt: str, query: str) -> dict:
    response = await ollama.AsyncClient().chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        options={
            "temperature": 0.2,
            "top_p": 0.8,
            "num_predict": 220,
        },
    )

    answer = (response or {}).get("message", {}).get("content", "").strip()
    return {"answer": answer or "I don't know"}


async def google_model(prompt: str, query: str) -> dict:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    combined_prompt = f"{prompt}\n\nUser query:\n{query}"

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=combined_prompt,
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

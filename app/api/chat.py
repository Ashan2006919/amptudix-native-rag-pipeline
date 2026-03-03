from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatBase
from app.core.database import query_rag
from app.agents.llm import get_answers

chat_router = APIRouter(prefix="/chat")

session_db = {}


@chat_router.post("")
async def chat_with_rag(request: ChatBase):
    try:
        results = await query_rag(request.query, 6)
        documents = (results or {}).get("documents") or []
        raw_chunks = documents[0] if documents else []

        deduped_chunks = []
        seen = set()
        for chunk in raw_chunks:
            normalized = " ".join(str(chunk).split()).lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped_chunks.append(str(chunk).strip())

        max_chunks = 4
        max_total_chars = 2500
        selected_chunks = []
        total_chars = 0
        for chunk in deduped_chunks[:max_chunks]:
            next_total = total_chars + len(chunk)
            if next_total > max_total_chars:
                break
            selected_chunks.append(chunk)
            total_chars = next_total

        context = "\n\n".join(selected_chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not context:
        return {"answer": "I don't know", "sources": []}

    system_prompt = f"""
    You are Amptudix AI.

    Task: Answer the user's query using ONLY the provided context.

    Rules:
    1) Be concise and focused on the exact query intent.
    2) Do not include jokes, filler, or meta commentary.
    4) If the answer isn't explicitly in the context, say you don't know. Do not confuse different entities (e.g., do not confuse the creator with the creation).  
    5) Prefer 3-6 bullet points when listing facts.
    6) If numbers are present in context, preserve them exactly.

    CONTEXT DATA:
    {context}
    """

    metadatas = (results or {}).get("metadatas") or []
    metadata_items = metadatas[0] if metadatas else []
    sources = list(
        {
            source
            for item in metadata_items
            for source in [item.get("source")]
            if isinstance(source, str) and source
        }
    )
    sources_header = ", ".join(sources)

    return StreamingResponse(
        get_answers(prompt=system_prompt, query=request.query, history=session_db),
        media_type="text/plain",
        headers={
            "X-Sources": sources_header,
            "Access-Control-Expose-Header": "X-Sources",
        },
    )

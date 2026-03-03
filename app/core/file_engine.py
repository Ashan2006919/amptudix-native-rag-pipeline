import os
from typing import List, Optional
import fitz
from app.core.config import settings


async def extract_text_from_file(
    folder_path: str, filename: Optional[str] = None
) -> List[dict]:
    raw_chunks = []

    files_to_process = (
        [filename]
        if filename
        else [
            f for f in os.listdir(folder_path) if f.lower().endswith((".pdf", ".txt"))
        ]
    )

    for file in files_to_process:
        text = ""
        file_path = os.path.join(settings.DATA_FOLDER, file)

        if file.lower().endswith(".pdf"):
            with fitz.open(file_path) as doc:
                for page in doc:
                    blocks = page.get_text("blocks")

                    for b in blocks:
                        text += b[4] + "\n"

        elif file.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as doc:
                text = doc.read()

        raw_chunks.append({"text": text, "filename": file})

    return raw_chunks


async def recursive_split(
    chunk_size: int, chunk_overlap: int, filename: Optional[str] = None
) -> List[dict]:
    raw_chunks = await extract_text_from_file(
        folder_path=settings.DATA_FOLDER, filename=filename
    )
    chunk_dict = []

    for chunk in raw_chunks:
        final_chunks = []
        text = chunk.get("text", "")
        if not text:
            continue
        paragraph = text.split("\n\n")

        current_chunk = ""
        for para in paragraph:
            para = para.strip()
            if not para or len(para) < 5:
                continue

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            elif len(para) <= chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())

                overlap_text = (
                    current_chunk[-(chunk_overlap):]
                    if len(current_chunk) > chunk_overlap
                    else ""
                )
                current_chunk += overlap_text + para + "\n\n"

            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())

                start = 0
                while start < len(para):
                    end = start + chunk_size

                    sub_chunk = para[start:end]
                    final_chunks.append(sub_chunk.strip())

                    start += chunk_size - chunk_overlap

        if current_chunk:
            final_chunks.append(current_chunk)

        chunk_dict.append({"chunk": final_chunks, "filename": chunk.get("filename")})

    return chunk_dict

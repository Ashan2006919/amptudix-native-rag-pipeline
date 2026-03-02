import os
from typing import List
import fitz
import ollama
from uuid import uuid4
from app.core.config import settings


def extract_from_pdf(folder_path) -> List[dict]:
    raw_chunks = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            text = ""
            file_path = os.path.join(folder_path, file)
            with fitz.open(file_path) as doc:
                for page in doc:
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        text += b[4] + "\n"

            raw_chunks.append({"text": text, "filename": file})

    return raw_chunks


def recursive_split(chunk_size: int, chunk_overlap: int) -> List[dict]:
    raw_chunks = extract_from_pdf(settings.DATA_FOLDER)
    chunk_dict = []

    for chunk in raw_chunks:
        chunks = []
        text = chunk.get("text")
        if not text:
            continue
        paragraph = text.split("\n\n")

        current_chunk = ""
        for para in paragraph:

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                overlap_text = (
                    current_chunk[-(chunk_overlap):]
                    if len(current_chunk) > chunk_overlap
                    else ""
                )
                current_chunk += overlap_text + para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk)

        chunk_dict.append({"chunk": chunks, "filename": chunk.get("filename")})

    return chunk_dict

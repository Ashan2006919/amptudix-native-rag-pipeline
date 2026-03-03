import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.core.config import settings
from app.core.database import process_one_file

file_router = APIRouter(prefix="/file")

background_tasks = BackgroundTasks()


@file_router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,  # <-- Inject it here
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # 1. Ensure directory exists
    os.makedirs(settings.DATA_FOLDER, exist_ok=True)
    destination_path = os.path.join(settings.DATA_FOLDER, file.filename)

    # 2. Save the file
    with open(destination_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Add the task (FastAPI handles the execution after sending the response)
    background_tasks.add_task(process_one_file, file.filename)

    return {"status": "File uploaded! Indexing has started in the background."}

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.chat import chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Amptudix Native Rag", version="0.1.0", lifespan=lifespan)

app.include_router(chat_router, tags=["Chat"])

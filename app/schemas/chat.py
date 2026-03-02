from sqlmodel import SQLModel


class ChatBase(SQLModel):
    query: str

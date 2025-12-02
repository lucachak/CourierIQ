from fastapi import APIRouter, Path
from pydantic import BaseModel

user = APIRouter(prefix="/user", tags=["User"])


class User(BaseModel):
    user_id: int
    email: str
    auth_token: str

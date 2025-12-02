from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str]

    class Config:
        orm_mode = True


class AutomationRequest(BaseModel):
    name: str
    description: Optional[str] = None
    interval_minutes: int   # tempo de execução automática

class AutomationResponse(BaseModel):
    task_id: int
    name: str
    interval_minutes: int
    status: str = "scheduled"

    class Config:
        orm_mode = True

class SystemStatus(BaseModel):
    api_status: str
    ml_engine: str
    active_tasks: int

from fastapi import APIRouter, HTTPException

from src.api.schemas import UserCreate, UserLogin, UserResponse
from src.api.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])
service = UserService()


@router.post("/register", response_model=UserResponse)
def register_user(data: UserCreate):
    return service.create_user(data)


@router.post("/login")
def login_user(data: UserLogin):
    return service.login_user(data)


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    user = service.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user

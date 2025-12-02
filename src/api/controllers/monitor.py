from fastapi import APIRouter
from src.api.schemas import SystemStatus

router = APIRouter()

@router.get("/status", response_model=SystemStatus)
def get_system_status():
    """Retorna informações do estado geral do sistema."""
    return SystemStatus(
        api_status="online",
        ml_engine="ready",
        active_tasks=5  # placeholder
    )

@router.get("/ping")
def ping():
    """Health-check simples."""
    return {"status": "ok"}

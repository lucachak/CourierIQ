from fastapi import APIRouter
from src.api.schemas import AutomationRequest, AutomationResponse
from src.api.services.automation_service import AutomationService

router = APIRouter()
service = AutomationService()

@router.post("/task", response_model=AutomationResponse)
def create_task(data: AutomationRequest):
    """Cria uma automação (ex.: recalcular rota a cada X minutos)."""
    return service.create_task(data)

@router.get("/tasks")
def list_tasks():
    """Lista automações registradas."""
    return service.list_tasks()

@router.delete("/task/{task_id}")
def delete_task(task_id: int):
    """Apaga uma automação existente."""
    return service.delete_task(task_id)

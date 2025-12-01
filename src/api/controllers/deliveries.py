from typing import List

from fastapi import APIRouter, Path
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/deliveries", tags=["Deliveries"])

from pydantic import BaseModel


class Delivery(BaseModel):
    id: int
    address: str
    status: str


@router.get("/", response_model=List[Delivery])
async def list_deliveries():
    """
    List all deliveries (mock)
    """
    return [
        {"id": 1, "address": "123 Main St", "status": "pending"},
        {"id": 2, "address": "456 Oak St", "status": "delivered"},
    ]


@router.post("/", response_model=Delivery)
async def create_delivery():
    """
    Create a delivery (mock)
    """
    return {"id": 3, "address": "789 Pine St", "status": "pending"}


@router.get("/{delivery_id}", response_model=Delivery)
async def get_delivery(delivery_id: int = Path(...)):
    """
    Get delivery details by ID (mock)
    """
    return {"id": delivery_id, "address": f"Address {delivery_id}", "status": "pending"}

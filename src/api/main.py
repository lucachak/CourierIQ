import logging

from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse

from src.api.controllers.deliveries import router as deliveries_router
from src.api.controllers.routes import router as routes_router
from src.utils.config import load_settings

config = load_settings()

logging.basicConfig(
    level=logging.DEBUG if config.app.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("courieriq")

app = FastAPI(
    title="CourierIQ API",
    description="🚀 Minimal & visual FastAPI template with config & logger",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(deliveries_router)
app.include_router(routes_router)


# -------------------------------
# Health & Config
# -------------------------------
@app.get("/health", tags=["Health"])
async def healthcheck():
    return {"status": "ok", "debug": config.app.DEBUG, "version": "1.0.0"}


@app.get("/config", tags=["Visual"])
async def show_config():
    return {
        "app": {
            "LOG_LEVEL": config.app.LOG_LEVEL,
            "MAX_REQUESTS_PER_MINUTE": config.app.MAX_REQUESTS_PER_MINUTE,
            "DEBUG": config.app.DEBUG,
        },
        "monitor": {
            "SCAN_INTERVAL": config.monitor.SCAN_INTERVAL,
            "MAX_CONCURRENT_SCANS": config.monitor.MAX_CONCURRENT_SCANS,
        },
        "automation": {
            "ENABLE_AUTOMATION": config.automation.ENABLE_AUTOMATION,
            "THREAT_SCORE_THRESHOLD": config.automation.THREAT_SCORE_THRESHOLD,
        },
    }


# -------------------------------
# Routes / Optimizer
# -------------------------------
@app.post("/routes/optimize", tags=["Routes"])
async def optimize_route():
    return {"route": "mock optimized"}


@app.get("/routes/{route_id}", tags=["Routes"])
async def get_route(route_id: int = Path(...)):
    return {"route_id": route_id, "status": "mock"}


@app.get("/routes/history", tags=["Routes"])
async def get_route_history():
    return {"history": []}


# -------------------------------
# Users / Couriers
# -------------------------------
@app.get("/users", tags=["Users"])
async def list_users():
    return {"users": []}


@app.get("/users/{user_id}", tags=["Users"])
async def get_user(user_id: int = Path(...)):
    return {"user_id": user_id, "status": "mock"}


@app.post("/users", tags=["Users"])
async def create_user():
    return {"status": "mock created"}


# -------------------------------
# Monitoring
# -------------------------------
@app.get("/monitor/status", tags=["Monitor"])
async def monitor_status():
    return {"monitor": "running"}


@app.get("/logs", tags=["Monitor"])
async def get_logs():
    return {"logs": []}


# -------------------------------
# Automation / AI
# -------------------------------
@app.post("/automation/run", tags=["Automation"])
async def run_automation():
    return {"automation": "started"}


@app.get("/automation/status", tags=["Automation"])
async def automation_status():
    return {"status": "idle"}


# -------------------------------
# Startup / Shutdown
# -------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("📦 CourierIQ API is starting up...")
    logger.info(f"Debug mode: {config.app.DEBUG}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 CourierIQ API is shutting down...")

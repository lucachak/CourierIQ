import logging

from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse

from src.api.controllers.automation import router as automation_router
from src.api.controllers.deliveries import router as deliveries_router
from src.api.controllers.monitor import router as monitor_router
from src.api.controllers.routes import router as routes_router
from src.api.controllers.users import router as user_router
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
app.include_router(automation_router)
app.include_router(monitor_router)
app.include_router(user_router)


@app.get("/health", tags=["Health"])
async def health_check():
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


@app.get("/logs", tags=["Monitor"])
async def get_logs():
    return {"logs": []}


@app.on_event("startup")
async def startup_event():
    logger.info("📦 CourierIQ API is starting up...")
    logger.info(f"Debug mode: {config.app.DEBUG}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 CourierIQ API is shutting down...")

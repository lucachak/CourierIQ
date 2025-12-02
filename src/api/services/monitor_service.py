class MonitorService:

    def get_status(self):
        return {
            "api_status": "online",
            "ml_engine": "ready",
            "active_tasks": 0
        }

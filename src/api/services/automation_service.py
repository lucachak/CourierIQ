class AutomationService:

    def __init__(self):

        # Simulação de banco de dados em memória
        self.tasks = {}
        self.next_id = 1

    def create_task(self, data):
        task_id = self.next_id
        self.tasks[task_id] = {
            "name": data.name,
            "description": data.description,
            "interval_minutes": data.interval_minutes,
            "status": "scheduled"
        }
        self.next_id += 1
        return {"task_id": task_id, **self.tasks[task_id]}

    def list_tasks(self):
        return self.tasks

    def delete_task(self, task_id):
        if task_id in self.tasks:
            del self.tasks[task_id]
            return {"deleted": True}
        return {"deleted": False}

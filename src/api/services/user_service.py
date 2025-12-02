class UserService:

    def __init__(self):
        self.users = {}
        self.next_id = 1

    def create_user(self, data):
        uid = self.next_id
        self.users[uid] = {
            "id": uid,
            "email": data.email,
            "full_name": data.full_name
        }
        self.next_id += 1
        return self.users[uid]

    def login_user(self, data):
        for user in self.users.values():
            if user["email"] == data.email:
                return {"message": "Login successful"}  
        return {"error": "Invalid credentials"}

    def get_user(self, user_id):
        return self.users.get(user_id)

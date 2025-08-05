
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv


load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "ragchatbot")
COLLECTION_NAME = "chat_logs"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
logs_collection = db[COLLECTION_NAME]

def log_chat(user_input: str, bot_response: str, email: str = None):
    log_data = {
        "timestamp": datetime.utcnow(),
        "user_input": user_input,
        "bot_response": bot_response,
        "email": email
    }
    logs_collection.insert_one(log_data)

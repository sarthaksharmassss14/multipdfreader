from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import certifi

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "ragchatbot")
COLLECTION_NAME = "chat_logs"

# Initialize MongoDB client 
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
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
summary_collection = db["chat_summaries"]

def log_summary(email: str, summary: str, intent: str, conversation: list):
    data = {
        "email": email,
        "timestamp": datetime.utcnow(),
        "summary": summary,
        "intent": intent,
        "conversation": conversation  # optional: full Q&A
    }
    summary_collection.insert_one(data)

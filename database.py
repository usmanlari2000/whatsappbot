from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")

mongo = MongoClient(MONGODB_URI)

mongo = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=100000,
    connectTimeoutMS=100000      
)

db = mongo[MONGODB_DB]

employees_col = db["employees"]
sessions_col = db["sessions"]
embeddings_col = db["embeddings"]


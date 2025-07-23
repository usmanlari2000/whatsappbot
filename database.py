from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")

mongo = MongoClient(MONGODB_URI)
db = mongo[MONGODB_DB]

employees_col = db["employees"]
sessions_col = db["sessions"]
embeddings_col = db["embeddings"]

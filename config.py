from dotenv import load_dotenv
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

REQUIRED_VARS = ["PORT", "OPENAI_API_KEY", "MONGODB_URI", "MONGODB_DB"]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]

if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

PORT = os.getenv("PORT")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

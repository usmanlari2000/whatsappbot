from dotenv import load_dotenv
import os
from openai import OpenAI
from twilio.request_validator import RequestValidator
from sentence_transformers import SentenceTransformer

load_dotenv()

REQUIRED_VARS = ["OPENAI_API_KEY", "TWILIO_AUTH_TOKEN", "MONGODB_URI", "MONGODB_DB"]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]

if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)
validator = RequestValidator(TWILIO_AUTH_TOKEN)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

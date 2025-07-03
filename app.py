from dotenv import load_dotenv
from flask import Flask, request
import os
import numpy as np
from pymongo import MongoClient
from twilio.twiml.messaging_response import MessagingResponse
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from functools import lru_cache

load_dotenv()

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["acme"]
employees_collection = db["employees"]
sessions_collection = db["sessions"]
embeddings_collection = db["embeddings"]

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    user_input = request.values.get("Body", "").strip()
    user_phone_number = request.values.get("From", "").replace("whatsapp:", "")
    current_time = time.time()

    if not employees_collection.find_one({"phone_number": user_phone_number}):
        return _reply("Sorry, your phone number isn't registered with any employee.")

    if len(user_input.split()) > 200:
        return _reply("Your message is too long.")

    session = sessions_collection.find_one({"phone_number": user_phone_number})

    if not session:
        session = {"phone_number": user_phone_number, "history": [], "last_seen": current_time}
        sessions_collection.insert_one(session)
    else:
        sessions_collection.update_one(
            {"phone_number": user_phone_number},
            {"$set": {"last_seen": current_time}}
        )

    documents = list(embeddings_collection.find({}, {"_id": 0}))
    doc_texts = [doc["text"] for doc in documents]
    doc_embeddings = np.array([doc["embedding"] for doc in documents], dtype=np.float32)

    model = get_model()
    embedded_query = model.encode(user_input, normalize_embeddings=True)
    scores = np.dot(doc_embeddings, embedded_query)
    top_indices = np.argsort(scores)[-3:][::-1]
    top_matches = [" ".join(doc_texts[i].split()[:400]) for i in top_indices]
    context = "\n------------\n".join(top_matches)

    system_prompt = (
        "You are an assistant responsible for answering employee queries at the Punjab Information Technology Board (PITB). "
        "Only answer queries that fall within your responsibilities. "
        "Use only the provided context to answer. "
        "If a query is relevant to your role but the provided context is insufficient to answer it properly, inform the user that someone from HR will contact them soon. "
        "Be as concise as possible."
    )

    history = session.get("history", [])[-6:]
    text_history = "\n".join(f"{m['role']}: {m['content']}" for m in history if m["role"] in {"user", "assistant"})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "system", "content": f"History:\n{text_history}"},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=1,
    )
    answer = response.choices[0].message.content

    new_entries = [
        {"role": "user", "content": user_input, "timestamp": current_time},
        {"role": "assistant", "content": answer, "timestamp": time.time()}
    ]
    sessions_collection.update_one(
        {"phone_number": user_phone_number},
        {"$push": {"history": {"$each": new_entries}}}
    )

    return _reply(answer)

def _reply(text):
    twilio_response = MessagingResponse()
    twilio_response.message(text)
    return str(twilio_response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

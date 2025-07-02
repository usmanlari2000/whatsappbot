from dotenv import load_dotenv
from flask import Flask, request
import os
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from pymongo import MongoClient
from twilio.twiml.messaging_response import MessagingResponse
import time
from openai import OpenAI

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("minilm_embeddings.json", "r") as f:
    documents = json.load(f)

doc_texts = [doc["text"] for doc in documents]
doc_embeddings = np.array([doc["embedding"] for doc in documents])

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["acme"]
employees_collection = db["employees"]
sessions_collection = db["sessions"]

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    user_input = request.values.get("Body", "").strip()
    user_phone_number = request.values.get("From", "").replace("whatsapp:", "")
    current_time = time.time()

    if not employees_collection.find_one({"phone_number": user_phone_number}):
        twilio_response = MessagingResponse()
        twilio_response.message(
            "Sorry, your phone number isn't registered to any employee. Please try contacting me using a registered phone number."
        )
        return str(twilio_response)

    if len(user_input.split()) > 200:
        twilio_response = MessagingResponse()
        twilio_response.message("Your message is too long.")
        return str(twilio_response)

    # Fetch or initialize session
    session = sessions_collection.find_one({"phone_number": user_phone_number})
    if not session:
        session = {"phone_number": user_phone_number, "history": [], "last_seen": current_time}
        sessions_collection.insert_one(session)
    else:
        sessions_collection.update_one(
            {"phone_number": user_phone_number},
            {"$set": {"last_seen": current_time}}
        )

    embedded_query = model.encode(user_input, normalize_embeddings=True)
    scores = np.dot(doc_embeddings, embedded_query)
    top_indices = np.argsort(scores)[-3:][::-1]
    top_matches = [" ".join(doc_texts[i].split()[:400]) for i in top_indices]
    combined_context = "\n------------\n".join(top_matches)

    system_prompt = (
        "You are an assistant responsible for answering employee queries at the Punjab Information Technology Board (PITB). "
        "Only answer queries that fall within your responsibilities. "
        "Use only the provided context to answer. "
        "If a query is relevant to your role but the provided context is insufficient to answer it properly, inform the user that someone from HR will contact them soon. "
        "Be as concise as possible."
    )

    history = session.get("history", [])[-6:]
    text_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history if msg["role"] in {"user", "assistant"}
    )

    message_history = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{combined_context}"},
        {"role": "system", "content": f"History:\n{text_history}"},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message_history,
        temperature=0.3,
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

    twilio_response = MessagingResponse()
    twilio_response.message(answer)
    return str(twilio_response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)

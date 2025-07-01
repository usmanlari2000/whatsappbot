from dotenv import load_dotenv
from flask import Flask, request
import os
from sentence_transformers import SentenceTransformer
import json
import numpy as np
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

with open("employees.json", "r") as f:
    employees = json.load(f)

authorized_phone_numbers = {emp["phone_number"] for emp in employees}

def load_sessions():
    if not os.path.exists("sessions.json"):
        return {}
    with open("sessions.json", "r") as f:
        return json.load(f)

def save_sessions(sessions):
    with open("sessions.json", "w") as f:
        json.dump(sessions, f, indent=2)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    user_input = request.values.get("Body", "").strip()
    user_phone_number = request.values.get("From", "").replace("whatsapp:", "")
    
    current_time = time.time()

    if user_phone_number not in authorized_phone_numbers:
        twilio_response = MessagingResponse()
        twilio_response.message(
            "Sorry, your phone number isn't registered to any employee. Please try messaging from a registered phone number."
        )
        
        return str(twilio_response)

    sessions = load_sessions()

    if user_phone_number not in sessions:
        sessions[user_phone_number] = {"history": [], "last_seen": current_time}

    sessions[user_phone_number]["last_seen"] = current_time

    embedded_query = model.encode(user_input, normalize_embeddings=True)
    scores = np.dot(doc_embeddings, embedded_query)
    best_match_index = int(np.argmax(scores))
    best_match = doc_texts[best_match_index]

    system_prompt = (
        "You are an HR assistant at the Punjab Information Technology Board (PITB). Only answer queries related to your job description. Use only the provided context to answer. Be as concise as possible."
    )

    message_history = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{best_match}"},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    sessions[user_phone_number]["history"].append({
        "role": "user",
        "content": user_input,
        "timestamp": current_time
    })
    sessions[user_phone_number]["history"].append({
        "role": "assistant",
        "content": answer,
        "timestamp": time.time()
    })

    save_sessions(sessions)

    twilio_response = MessagingResponse()
    twilio_response.message(answer)
    return str(twilio_response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)

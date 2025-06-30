from flask import Flask, request
import os
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from twilio.twiml.messaging_response import MessagingResponse
import time
import openai

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

with open("bge_embeddings.json", "r") as f:
    documents = json.load(f)

doc_texts = [doc["text"] for doc in documents]
doc_embeddings = np.array([doc["embedding"] for doc in documents])

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
    user_id = request.values.get("From", "")
    current_time = time.time()

    sessions = load_sessions()

    if user_id in sessions:
        last_seen = sessions[user_id]["last_seen"]
        if current_time - last_seen > 3600:
            del sessions[user_id]

    if user_id not in sessions:
        sessions[user_id] = {"history": [], "last_seen": current_time}

    sessions[user_id]["last_seen"] = current_time

    embedded_query = model.encode(
        ["Represent the query for retrieval: " + user_input],
        normalize_embeddings=True
    )[0]

    scores = np.dot(doc_embeddings, embedded_query)
    best_match_index = int(np.argmax(scores))
    best_match = doc_texts[best_match_index]

    system_prompt = "You are an HR assistant at the Punjab Information Technology Board (PITB). Only answer PITB HR-related questions. Use only the provided context to respond. Be clear and concise."

    short_history = sessions[user_id]["history"][-10:]

    message_history = [
        {"role": "system", "content": system_prompt},
        *short_history,
        {"role": "system", "content": f"Context:\n{best_match}"},
        {"role": "user", "content": user_input}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=message_history,
        temperature=0.3,
    )

    answer = response["choices"][0]["message"]["content"]

    sessions[user_id]["history"].append({"role": "user", "content": user_input})
    sessions[user_id]["history"].append({"role": "assistant", "content": answer})
    save_sessions(sessions)

    twilio_response = MessagingResponse()
    twilio_response.message(answer)
    return str(twilio_response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)

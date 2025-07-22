from flask import Flask, request
from twilio_auth import is_valid_twilio_request
import time
from database import employees_col, sessions_col
from session import get_or_create_session, update_session
from config import client, model, validator
import numpy as np
from embeddings import doc_texts, doc_embeddings
from chatgpt import build_messages, get_chat_response
from utils import _reply

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    if not is_valid_twilio_request(request):
        return "Forbidden", 403

    user_input = request.values.get("Body", "").strip()
    user_phone_number = request.values.get("From", "").replace("whatsapp:", "")
    current_time = time.time()

    if not employees_col.find_one({"phone_number": user_phone_number}):
        return _reply("Sorry, I can't help you. Your phone number isn't registered to any employee. Please try again using a registered number.")

    if len(user_input.split()) > 200:
        return _reply("Your message is too long.")

    session = get_or_create_session(user_phone_number, current_time)

    embedded_query = model.encode(user_input, normalize_embeddings=True)
    match_index = np.argmax(np.dot(doc_embeddings, embedded_query))
    context = doc_texts[match_index]

    messages = build_messages(user_input, session.get("history", []), context)
    answer = get_chat_response(messages)

    update_session(user_phone_number, session, user_input, answer, current_time)
    return _reply(answer)

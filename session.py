from database import sessions_col
import time

def get_or_create_session(phone, now):
    session = sessions_col.find_one({"_id": phone}) or {
        "_id": phone,
        "history": [],
        "last_seen": now
    }
    session["last_seen"] = now
    
    return session

def update_session(phone, session, user_input, answer, now):
    session["history"].extend([
        {"role": "user", "content": user_input, "timestamp": now},
        {"role": "assistant", "content": answer, "timestamp": time.time()}
    ])
    sessions_col.replace_one({"_id": phone}, session, upsert=True)

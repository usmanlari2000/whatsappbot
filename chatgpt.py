from config import client

def build_messages(user_input, history, context):
    system_prompt = (
        "You are an assistant responsible for answering employee queries at the Punjab Information Technology Board (PITB). Only answer queries that fall within your defined responsibilities. Use only the provided context to answer. Only use the context if needed. Keep your response below 1200 characters."
    )

    history_text = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-10:] if m["role"] in {"user", "assistant"})

    return [
        {"role": "system", "content": f"System prompt:\n{system_prompt}"},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "system", "content": f"History:\n{history_text}"},
        {"role": "user", "content": f"New message:\n{user_input}"}
    ]

def get_chat_response(messages):
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=1,
        )
        return res.choices[0].message.content[:1200]

    except Exception as e:
        print(f"OpenAI error: {e}")
        
        return "Sorry, I couldn't generate a response. Please try again later."

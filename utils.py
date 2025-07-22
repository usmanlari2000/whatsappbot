from twilio.twiml.messaging_response import MessagingResponse

def _reply(text):
    res = MessagingResponse()
    res.message(text)

    return str(res)

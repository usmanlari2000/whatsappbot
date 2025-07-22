from config import validator

def is_valid_twilio_request(request):
    signature = request.headers.get("X-Twilio-Signature", "")
    url = request.url
    post_vars = request.form.to_dict()
    
    return validator.validate(url, post_vars, signature)

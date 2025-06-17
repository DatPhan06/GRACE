import requests
import json

API_KEY = "AIzaSyDy2rfcehMxGfFyDA7mmQAupcD-rKUjPvc"  # Thay bằng API key của bạn

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
headers = {
    "Content-Type": "application/json"
}
data = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Explain how AI works in a few words"
                }
            ]
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print("Status code:", response.status_code)
print("Response:", response.text)
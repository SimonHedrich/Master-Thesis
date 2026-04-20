import requests

API_KEY = "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"

payload = {
    "model": "google/gemini-3.1-flash-image-preview", # You MUST switch to the Flash model
    "messages": [
        {"role": "user", "content": "A futuristic cityscape at night"}
    ],
    "modalities": ["image", "text"], 
    "image_config": {
        "aspect_ratio": "16:9", 
        "image_size": "0.5K"      # The 512px tier
    }
}

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json=payload
)

print(response.json())
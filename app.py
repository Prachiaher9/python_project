
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import re
from langdetect import detect
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
print("TOKEN:", HUGGINGFACE_API_TOKEN)
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}


def query_hf_model(model_name: str, text: str):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()  # raises error if request failed
    return response.json()

class TextInput(BaseModel):
    name: str
    review: str

reviews = []

def is_hinglish(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text)
    hindi_chars = re.findall(r"[\u0900-\u097F]", text)

    if words and hindi_chars:
        return True

    try:
        lang = detect(text)
        if lang == "hi" and any(char.isascii() for char in text):
            return True
    except:
        pass

    return False

@app.post("/add-review")
def add_review(data: TextInput):
    text = data.review
    if is_hinglish(text):
        model_name = "ganeshkharad/gk-hinglish-sentiment"
    else:
        model_name = "tabularisai/multilingual-sentiment-analysis"

    result = query_hf_model(model_name, text)

    sentiment = "UNKNOWN"
    if (
        isinstance(result, list)
        and len(result) > 0
        and isinstance(result[0], list)
        and len(result[0]) > 0
    ):
        sentiment = result[0][0]['label']

    reviews.append({
        "name": data.name,
        "review": text,
        "sentiment": sentiment
    })

    return {"message": "Review added successfully"}


@app.get("/get-reviews")
def get_reviews():
    return reviews
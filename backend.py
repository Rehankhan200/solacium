
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# FastAPI app
app = FastAPI(title="Student Mental Health Support")

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Predefined keyword responses
keyword_responses = {
    "bullied": "We understand you're facing bullying. Stay strong, you are not alone!",
    "harassed": "If you are being harassed, it's okay to reach out to trusted people for help.",
    "stressed": "Take a deep breath. Try to relax and prioritize your tasks.",
    "assignment": "Don't worry about late assignments, communicate with your teacher.",
    "introvert": "Being introverted is fine. Embrace your strengths and pace yourself.",
}

# Input model
class StudentMessage(BaseModel):
    message: str

# Initialize AI emotion model (GoEmotions / DistilBERT)
emotion_analyzer = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/analyze")
def analyze_message(data: StudentMessage):
    text = data.message.lower()
    response = "Sorry, I couldn't understand completely. But we are here to help!"
    
    # Keyword detection
    for key, val in keyword_responses.items():
        if key in text:
            response = val
            break
    
    # AI emotion detection (for demo/future)
    ai_result = emotion_analyzer(data.message)
    ai_emotion = ai_result[0]["label"]
    
    return {"response": response, "ai_emotion": ai_emotion}

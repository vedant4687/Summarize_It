import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "vsd4687/T5-Summarizer"

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

app = FastAPI(title="Summarize It", version="1.0.0")
templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def summarize_dialogue(dialogue: str) -> str:
    cleaned = clean_text(dialogue)
    inputs = tokenizer(cleaned, return_tensors="pt", max_length=256, truncation=True)
    tokens = model.generate(**inputs, max_length=96, num_beams=2)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}
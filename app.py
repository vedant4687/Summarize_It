from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Summarize It", description="Text summarization app using ONNX T5 model", version="1.0.0")

MODEL_NAME = "vsd4687/T5-Summarizer"

# Load tokenizer & ONNX model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

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

@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
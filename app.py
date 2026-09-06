from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Summarize It", description="Text summarization app using T5 model", version="1.0.0")

MODEL_NAME = "vsd4687/T5-Summarizer"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model using low CPU memory usage
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, 
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)

device = torch.device("cpu")
model.to(device)
model.eval()  # Disable training layers to reduce memory overhead

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_text(dialogue)
    
    inputs = tokenizer(
        dialogue, 
        padding="max_length", 
        truncation=True, 
        max_length=256,  # Reduced max token length to save RAM
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():  # Prevents storing gradient history in RAM
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=96,
            num_beams=2,     # Reduced beam count from 4 to 2 to cut RAM usage during generation
            early_stopping=True
        )
    
    return tokenizer.decode(targets[0], skip_special_tokens=True)

@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
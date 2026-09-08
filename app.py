import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import onnxruntime as ort
from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Restrict thread allocation to lower memory overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

MODEL_NAME = "vsd4687/T5-Summarizer"
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    # Configure ONNX Runtime to prioritize low memory usage
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.enable_cpu_mem_arena = False
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    model = ORTModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, 
        provider="CPUExecutionProvider",
        session_options=sess_options
    )
    yield

app = FastAPI(title="Summarize It", version="1.0.0", lifespan=lifespan)
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
    tokens = model.generate(**inputs, max_length=96, num_beams=1)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


#initialize the FastAPI app
app= FastAPI(title= "Summarize It", description= "Text summarization app using T5 model", version= "1.0.0")

#model and tokenizer initialization
model= T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer= T5Tokenizer.from_pretrained("./saved_summary_model")

#device
if torch.cuda.is_available():
    device= torch.device("cuda")
else:
    device= torch.device("cpu")

model.to(device)

#Template setup
templates= Jinja2Templates(directory= ".")

#Input schema for the request body
class DialogueInput(BaseModel):
    dialogue: str
    
#Define clean data function
def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)  # Replace newlines with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text= re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text= text.strip()  # Remove leading and trailing whitespace
    return text

#Summarization function
def summarize_dialogue(dialogue: str)->str:
    dialogue = clean_text(dialogue)
    
    # 1. Tokenize input
    inputs = tokenizer(
        dialogue, 
        padding="max_length", 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # 2. Generate summary with mixed precision for RTX 3060 speedup
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            targets = model.generate(
                input_ids=inputs["input_ids"],            # Fixed variable name
                attention_mask=inputs["attention_mask"],  # Fixed variable name
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
    
    # 3. Decode output tokens
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary
    
    
#API Enpoints
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )
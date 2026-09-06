import re
import streamlit as st
import streamlit.components.v1 as components
from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Page configuration
st.set_page_config(page_title="Summarize It", layout="wide")

MODEL_NAME = "vsd4687/T5-Summarizer"

@st.cache_resource
def load_model():
    # Load tokenizer from t5-small to access spiece.model correctly
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    # Load fine-tuned ONNX model weights
    model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAME, provider="CPUExecutionProvider")
    return tokenizer, model

tokenizer, model = load_model()

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

# 1. Display Custom UI HTML File
with open("index.html", "r", encoding="utf-8") as f:
    html_code = f.read()

components.html(html_code, height=600, scrolling=True)

# 2. Streamlit Model Processing Controls
st.sidebar.title("Text Summarizer")
user_input = st.sidebar.text_area("Enter Content / Text:", height=220)

if st.sidebar.button("Generate Summary"):
    if user_input.strip():
        with st.sidebar.spinner("Generating summary..."):
            summary = summarize_dialogue(user_input)
            st.sidebar.subheader("Summary")
            st.sidebar.write(summary)
    else:
        st.sidebar.warning("Please enter the Content to summarize.")
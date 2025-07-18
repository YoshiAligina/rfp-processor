from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_NAME = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def chunk_text(text, max_tokens=4000, stride=200):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens - stride):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def predict_document_probability(text):
    chunks = chunk_text(text)
    probs = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=4096)
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=1)[0][1].item()
            probs.append(prob)
    return float(np.mean(probs)) if probs else 0.0

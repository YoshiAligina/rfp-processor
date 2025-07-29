from transformers import LongformerTokenizer, LongformerForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import re
import warnings
from typing import List, Tuple

# Suppress the transformer warning about uninitialized weights
warnings.filterwarnings("ignore", message="Some weights of LongformerForSequenceClassification were not initialized")
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")

# Also suppress transformers logging for cleaner output
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "allenai/longformer-base-4096"
MODEL_SAVE_PATH = "fine_tuned_longformer"
tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)

# Try to load fine-tuned model first, fallback to base model
if os.path.exists(MODEL_SAVE_PATH):
    print(f"[INFO] Loading fine-tuned model from {MODEL_SAVE_PATH}")
    model = LongformerForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
else:
    print(f"[INFO] Loading base model from {MODEL_NAME}")
    model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

class RFPDataset(Dataset):
    """Custom dataset for RFP text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 4096):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_training_data():
    """Prepare training data from historical decisions"""
    historical_df = load_historical_decisions()
    
    if len(historical_df) < 2:
        print("[INFO] Not enough historical data for training (need at least 2 examples)")
        return None, None
    
    texts = []
    labels = []
    
    for _, row in historical_df.iterrows():
        summary = row.get('summary', '')
        if summary and len(summary.strip()) > 20:  # Ensure meaningful content
            texts.append(summary)
            # Convert decision to binary label: Approved=1, Denied=0
            labels.append(1 if row['decision'] == 'Approved' else 0)
    
    if len(texts) < 2:
        print("[INFO] Not enough valid training examples")
        return None, None
    
    return texts, labels

def fine_tune_model(epochs: int = 3, batch_size: int = 2, learning_rate: float = 2e-5):
    """Fine-tune the Longformer model on RFP data"""
    global model
    
    print(f"[INFO] Starting fine-tuning process...")
    
    # Prepare data
    texts, labels = prepare_training_data()
    if texts is None:
        return False
    
    print(f"[INFO] Training on {len(texts)} examples")
    
    # Split data if we have enough examples
    if len(texts) >= 4:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
    else:
        # Use all data for training if we don't have enough for validation
        train_texts, train_labels = texts, labels
        val_texts, val_labels = [], []
    
    # Create datasets
    train_dataset = RFPDataset(train_texts, train_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataloader = None
    if val_texts:
        val_dataset = RFPDataset(val_texts, val_labels, tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        print(f"[INFO] Epoch {epoch + 1}/{epochs}")
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"[INFO] Average loss for epoch {epoch + 1}: {avg_loss:.4f}")
        
        # Validation if we have validation data
        if val_dataloader:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = correct / total
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"[INFO] Validation - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            model.train()
    
    # Save the fine-tuned model
    try:
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print(f"[INFO] Fine-tuned model saved to {MODEL_SAVE_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        return False

def chunk_text(text, max_tokens=4000, stride=500):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"[DEBUG] Total tokens in document: {len(tokens)}")
    chunks = []
    for i in range(0, len(tokens), max_tokens - stride):
        chunk = tokens[i:i + max_tokens]
        print(f"[DEBUG] Chunk {len(chunks)}: Tokens {i} to {i+len(chunk)}")
        chunks.append(tokenizer.decode(chunk))
    return chunks

def extract_rfp_features(text):
    """Extract key features from RFP text for similarity comparison"""
    if not text:
        return ""
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Extract key RFP-related terms and phrases
    key_patterns = [
        r'budget[:\s]+\$?[\d,]+',  # Budget information
        r'deadline[:\s]+[\w\s,]+',  # Deadline mentions
        r'project\s+duration[:\s]+[\w\s,]+',  # Project duration
        r'services?\s+required?[:\s]*[\w\s,]+',  # Required services
        r'experience\s+required?[:\s]*[\w\s,]+',  # Experience requirements
        r'qualifications?[:\s]*[\w\s,]+',  # Qualifications
        r'scope\s+of\s+work[:\s]*[\w\s,]+',  # Scope of work
        r'deliverables?[:\s]*[\w\s,]+',  # Deliverables
        r'location[:\s]*[\w\s,]+',  # Location
        r'industry[:\s]*[\w\s,]+',  # Industry
    ]
    
    # Extract matched patterns
    features = []
    for pattern in key_patterns:
        matches = re.findall(pattern, text)
        features.extend(matches)
    
    # Also include the first 500 characters as general context
    features.append(text[:500])
    
    return ' '.join(features)

def load_historical_decisions():
    """Load historical approved/denied decisions for learning"""
    csv_db = "rfp_db.csv"
    if not os.path.exists(csv_db):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_db)
        # Only return entries with actual decisions (not pending)
        return df[df['decision'].isin(['Approved', 'Denied'])].copy()
    except Exception:
        return pd.DataFrame()

def calculate_similarity_score(current_text, historical_df):
    """Calculate similarity between current document and historical decisions"""
    if historical_df.empty:
        return 0.5  # Neutral score if no historical data
    
    # Extract features from current document
    current_features = extract_rfp_features(current_text)
    if not current_features.strip():
        return 0.5
    
    # Extract features from historical documents
    approved_docs = historical_df[historical_df['decision'] == 'Approved']
    denied_docs = historical_df[historical_df['decision'] == 'Denied']
    
    if approved_docs.empty and denied_docs.empty:
        return 0.5
    
    # Prepare text data for vectorization
    all_texts = [current_features]
    labels = ['current']
    
    # Add approved documents
    for _, row in approved_docs.iterrows():
        summary = row.get('summary', '')
        if summary and len(summary.strip()) > 20:
            features = extract_rfp_features(summary)
            if features.strip():
                all_texts.append(features)
                labels.append('approved')
    
    # Add denied documents
    for _, row in denied_docs.iterrows():
        summary = row.get('summary', '')
        if summary and len(summary.strip()) > 20:
            features = extract_rfp_features(summary)
            if features.strip():
                all_texts.append(features)
                labels.append('denied')
    
    if len(all_texts) < 2:  # Need at least current + one historical
        return 0.5
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        current_vector = tfidf_matrix[0]
        similarities = cosine_similarity(current_vector, tfidf_matrix[1:]).flatten()
        
        # Separate approved and denied similarities
        approved_similarities = []
        denied_similarities = []
        
        for i, label in enumerate(labels[1:]):  # Skip 'current'
            if label == 'approved':
                approved_similarities.append(similarities[i])
            elif label == 'denied':
                denied_similarities.append(similarities[i])
        
        # Calculate average similarities
        avg_approved_sim = np.mean(approved_similarities) if approved_similarities else 0
        avg_denied_sim = np.mean(denied_similarities) if denied_similarities else 0
        
        # Convert similarity difference to probability
        # Higher similarity to approved docs = higher probability
        sim_diff = avg_approved_sim - avg_denied_sim
        
        # Normalize to 0-1 range using sigmoid-like function
        similarity_score = 1 / (1 + np.exp(-5 * sim_diff))
        
        return float(similarity_score)
        
    except Exception as e:
        print(f"[DEBUG] Error in similarity calculation: {e}")
        return 0.5

def predict_document_probability(text, auto_finetune=True):
    """Enhanced prediction combining fine-tuned model with historical learning"""
    global model
    
    print(f"[DEBUG] Starting enhanced prediction for document...")
    
    # Check if we should trigger auto fine-tuning
    if auto_finetune:
        historical_df = load_historical_decisions()
        # Auto fine-tune if we have enough new decisions and no fine-tuned model exists
        if (len(historical_df) >= 5 and not os.path.exists(MODEL_SAVE_PATH)) or \
           (len(historical_df) >= 10 and len(historical_df) % 10 == 0):  # Re-train every 10 new decisions
            print(f"[INFO] Triggering automatic fine-tuning with {len(historical_df)} historical decisions")
            fine_tune_success = fine_tune_model()
            if fine_tune_success:
                # Reload the fine-tuned model
                model = LongformerForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
                print("[INFO] Fine-tuned model reloaded successfully")
    
    # Get base model prediction
    chunks = chunk_text(text)
    base_probs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    for idx, chunk in enumerate(chunks):
        print(f"[DEBUG] Processing chunk {idx}: {chunk[:100]}...")  # First 100 chars
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=4096, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            print(f"[DEBUG] Logits for chunk {idx}: {logits}")
            prob = torch.softmax(logits, dim=1)[0][1].item()
            print(f"[DEBUG] Base model probability for chunk {idx}: {prob}")
            base_probs.append(prob)
    
    base_probability = float(np.mean(base_probs)) if base_probs else 0.5
    print(f"[DEBUG] Base model averaged probability: {base_probability}")
    
    # Get historical learning component
    historical_df = load_historical_decisions()
    similarity_score = calculate_similarity_score(text, historical_df)
    print(f"[DEBUG] Historical similarity score: {similarity_score}")
    
    # Adjust weights based on whether we have a fine-tuned model
    num_historical = len(historical_df)
    has_finetuned_model = os.path.exists(MODEL_SAVE_PATH)
    
    if has_finetuned_model:
        # With a fine-tuned model, we can rely more on the base model
        if num_historical >= 20:
            weight_base = 0.7
            weight_similarity = 0.3
        elif num_historical >= 10:
            weight_base = 0.6
            weight_similarity = 0.4
        else:
            weight_base = 0.5
            weight_similarity = 0.5
    else:
        # Without fine-tuning, rely more heavily on similarity learning
        if num_historical >= 10:
            weight_base = 0.2
            weight_similarity = 0.8
        elif num_historical >= 5:
            weight_base = 0.3
            weight_similarity = 0.7
        elif num_historical >= 2:
            weight_base = 0.4
            weight_similarity = 0.6
        else:
            weight_base = 0.5
            weight_similarity = 0.5
    
    final_probability = (weight_base * base_probability) + (weight_similarity * similarity_score)
    
    model_type = "fine-tuned" if has_finetuned_model else "base"
    print(f"[DEBUG] Final probability: {final_probability} (base: {base_probability}, similarity: {similarity_score}, weights: {weight_base}/{weight_similarity})")
    print(f"[DEBUG] Used {num_historical} historical decisions with {model_type} model")
    
    return float(final_probability)

def manual_fine_tune():
    """Manually trigger fine-tuning process"""
    print("[INFO] Manual fine-tuning triggered")
    return fine_tune_model()

def get_model_info():
    """Get information about the current model state"""
    historical_df = load_historical_decisions()
    has_finetuned = os.path.exists(MODEL_SAVE_PATH)
    
    return {
        'has_fine_tuned_model': has_finetuned,
        'historical_decisions': len(historical_df),
        'approved_count': len(historical_df[historical_df['decision'] == 'Approved']) if not historical_df.empty else 0,
        'denied_count': len(historical_df[historical_df['decision'] == 'Denied']) if not historical_df.empty else 0,
        'model_path': MODEL_SAVE_PATH if has_finetuned else MODEL_NAME
    }


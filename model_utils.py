from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import re

MODEL_NAME = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)
model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

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

def predict_document_probability(text):
    """Enhanced prediction combining base model with historical learning"""
    print(f"[DEBUG] Starting enhanced prediction for document...")
    
    # Get base model prediction
    chunks = chunk_text(text)
    base_probs = []
    for idx, chunk in enumerate(chunks):
        print(f"[DEBUG] Processing chunk {idx}: {chunk[:100]}...")  # First 100 chars
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=4096, padding="max_length")
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
    
    # Combine scores with weighted average
    # Give more weight to similarity if we have enough historical data
    num_historical = len(historical_df)
    if num_historical >= 5:
        # With sufficient data, weight similarity more heavily
        weight_base = 0.3
        weight_similarity = 0.7
    elif num_historical >= 2:
        # With some data, balance the weights
        weight_base = 0.5
        weight_similarity = 0.5
    else:
        # With little/no data, rely mostly on base model
        weight_base = 0.8
        weight_similarity = 0.2
    
    final_probability = (weight_base * base_probability) + (weight_similarity * similarity_score)
    
    print(f"[DEBUG] Final probability: {final_probability} (base: {base_probability}, similarity: {similarity_score}, weights: {weight_base}/{weight_similarity})")
    print(f"[DEBUG] Used {num_historical} historical decisions for learning")
    
    return float(final_probability)


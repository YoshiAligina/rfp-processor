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
TRAINING_TRACKER_FILE = "training_tracker.txt"  # Track last training count
tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)

# Training tracking functions
def get_last_training_count():
    """Get the number of entries when we last performed training."""
    if os.path.exists(TRAINING_TRACKER_FILE):
        try:
            with open(TRAINING_TRACKER_FILE, 'r') as f:
                return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            return 0
    return 0

def save_training_count(count):
    """Save the current entry count after training."""
    try:
        with open(TRAINING_TRACKER_FILE, 'w') as f:
            f.write(str(count))
        print(f"📊 [TRACKING] Saved training count: {count} entries")
    except Exception as e:
        print(f"⚠️ [TRACKING] Failed to save training count: {e}")

# Try to load fine-tuned model first, fallback to base model
if os.path.exists(MODEL_SAVE_PATH):
    print(f"[INFO] Loading fine-tuned model from {MODEL_SAVE_PATH}")
    model = LongformerForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
else:
    print(f"[INFO] Loading base model from {MODEL_NAME}")
    model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

class RFPDataset(Dataset):
    """
    Custom dataset class for RFP text classification using PyTorch Dataset interface.
    This class handles the tokenization and formatting of RFP documents for training
    the Longformer model. It converts raw text into tokenized input suitable for
    transformer models while maintaining the associated classification labels.
    """
    
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
    """
    Prepares training data from historical RFP decisions stored in the CSV database.
    This function loads historical decisions, filters for valid entries with meaningful
    content, and converts text summaries and decisions into format suitable for model training.
    Returns paired lists of text content and binary labels (1=Approved, 0=Denied).
    """
    print("[DEBUG] Loading historical decisions for training data preparation...")
    historical_df = load_historical_decisions()
    
    print(f"[DEBUG] Found {len(historical_df)} historical decisions")
    
    if len(historical_df) < 2:
        print(f"[WARNING] Not enough historical data for training (need at least 2 examples, found {len(historical_df)})")
        return None, None
    
    texts = []
    labels = []
    skipped_count = 0
    
    for idx, row in historical_df.iterrows():
        summary = row.get('summary', '')
        decision = row.get('decision', '')
        
        print(f"[DEBUG] Processing entry {idx}: decision={decision}, summary_length={len(summary) if summary else 0}")
        
        if summary and len(summary.strip()) > 20:  # Ensure meaningful content
            texts.append(summary)
            # Convert decision to binary label: Approved=1, Denied=0
            label = 1 if row['decision'] == 'Approved' else 0
            labels.append(label)
            print(f"[DEBUG] Added training example: label={label}, summary_preview='{summary[:50]}...'")
        else:
            skipped_count += 1
            print(f"[DEBUG] Skipped entry {idx}: insufficient summary content")
    
    print(f"[DEBUG] Training data preparation complete:")
    print(f"[DEBUG] - Valid examples: {len(texts)}")
    print(f"[DEBUG] - Skipped examples: {skipped_count}")
    print(f"[DEBUG] - Label distribution: {dict(zip(*np.unique(labels, return_counts=True))) if labels else 'No labels'}")
    
    if len(texts) < 2:
        print(f"[WARNING] Not enough valid training examples (need at least 2, found {len(texts)})")
        return None, None
    
    return texts, labels

def fine_tune_model(epochs: int = 2, batch_size: int = 1, learning_rate: float = 2e-5):
    """
    SIMPLIFIED Fine-tuning process for the RFP classification model.
    
    WHAT THIS FUNCTION DOES:
    This function customizes the pre-trained Longformer model to specifically understand
    RFP documents by training it on historical approval/denial decisions. It takes the
    generic language understanding of Longformer and teaches it to recognize patterns
    that lead to RFP approvals vs denials.
    
    THE FINE-TUNING PROCESS:
    1. Load historical RFP decisions (approved/denied examples)
    2. Convert text and decisions into training format
    3. Set up training parameters (optimizer, scheduler, device)
    4. Train the model through multiple epochs (complete passes through data)
    5. For each batch: make predictions, calculate error, update model weights
    6. Save the improved model for future use
    
    WHY FINE-TUNING WORKS:
    - Pre-trained model already understands language and context
    - Fine-tuning adapts this understanding to RFP-specific patterns
    - Uses relatively few examples to achieve domain-specific expertise
    - Much faster than training from scratch
    
    BACKGROUND TRAINING FEATURES:
    - Training continues even if you switch tabs or minimize the window
    - Progress is saved automatically after each epoch
    - Robust error handling prevents crashes
    - Clear logging shows training status
    
    PARAMETERS:
    - epochs: How many times to go through all training data (default: 2)
    - batch_size: How many examples to process at once (default: 1 for small datasets)
    - learning_rate: How aggressively to update model weights (default: 2e-5, conservative)
    """
    global model
    
    # BACKGROUND PROCESSING SETUP
    # These settings ensure training continues regardless of UI focus
    import sys
    import gc
    import time
    
    # Flush output to ensure progress messages appear immediately
    sys.stdout.flush()
    
    print(f"🚀 [BACKGROUND TRAINING] Starting robust fine-tuning...")
    print(f"💡 [INFO] Training will continue even if you switch tabs or minimize window")
    print(f"📝 [INFO] Progress will be automatically saved after each epoch")
    
    try:
        # STEP 1: PREPARE TRAINING DATA
        # Load historical RFP decisions and convert them to training format
        # This creates pairs of (text_summary, approval_decision) for learning
        texts, labels = prepare_training_data()
        if texts is None:
            print("❌ [ERROR] Not enough training data available")
            sys.stdout.flush()
            return False
    
        print(f"📊 [INFO] Training with {len(texts)} examples")
        label_counts = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"📈 [INFO] Data: {label_counts}")
        sys.stdout.flush()
        
        # STEP 2: DATA STRATEGY FOR SMALL DATASETS
        # SIMPLIFIED: Use ALL data for training (no validation split for small datasets)
        # Why: With limited RFP data, every example is valuable for learning patterns
        # Traditional ML splits data (80% train, 20% validation), but with <50 examples,
        # we prioritize learning over validation to maximize pattern recognition
        print(f"💡 [STRATEGY] Using all data for training (maximizes learning)")
        sys.stdout.flush()
        
        # STEP 3: SETUP COMPUTING ENVIRONMENT
        # Check if GPU is available for faster training (CUDA), otherwise use CPU
        # GPU can train ~10x faster, but CPU works fine for small datasets
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ [DEVICE] Using: {device}")
        sys.stdout.flush()
        
        # STEP 4: PREPARE DATA FOR PYTORCH TRAINING
        # Convert our text/label pairs into PyTorch format that the model can process
        # RFPDataset handles tokenization (converting text to numbers) and formatting
        # DataLoader manages batching and shuffling during training
        train_dataset = RFPDataset(texts, labels, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"📦 [DATA] Created {len(train_dataloader)} training batches")
        sys.stdout.flush()
        
        # STEP 5: CONFIGURE TRAINING PARAMETERS
        # Optimizer: AdamW is a sophisticated algorithm that decides how to update model weights
        # - learning_rate: How big steps to take when updating (2e-5 is conservative)
        # - weight_decay: Prevents overfitting by penalizing large weights
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_dataloader) * epochs
        
        # Scheduler: Gradually reduces learning rate during training for better convergence
        # Starts with warmup (gradual increase), then linear decay
        # This helps the model learn effectively without overshooting optimal weights
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
    
        # STEP 6: PREPARE MODEL FOR TRAINING
        # Move model to the computing device (GPU/CPU) and set to training mode
        # Training mode enables dropout and batch normalization for learning
        model.to(device)
        model.train()
        
        print(f"🏃 [TRAINING] Starting {epochs} epochs with {total_steps} total steps")
        print(f"🔄 [BACKGROUND] Training will continue in background - safe to switch tabs!")
        sys.stdout.flush()
        
        # STEP 7: THE MAIN TRAINING LOOP WITH BACKGROUND SUPPORT
        # This is where the actual learning happens through multiple epochs
        # Enhanced with robust background processing capabilities
        for epoch in range(epochs):
            epoch_loss = 0  # Track total loss for this epoch
            epoch_start_time = time.time()
            print(f"\n🔄 [EPOCH {epoch + 1}/{epochs}] Starting...")
            sys.stdout.flush()
            
            # Process each batch of training data
            for batch_idx, batch in enumerate(train_dataloader):
                batch_start_time = time.time()
                
                # SUBSTEP 7a: RESET GRADIENTS
                # Clear gradients from previous batch (PyTorch accumulates them by default)
                optimizer.zero_grad()
                
                # SUBSTEP 7b: MOVE DATA TO DEVICE
                # Transfer input data to GPU/CPU to match model location
                input_ids = batch['input_ids'].to(device)        # Tokenized text
                attention_mask = batch['attention_mask'].to(device)  # Which tokens to pay attention to
                batch_labels = batch['labels'].to(device)       # Correct answers (approved/denied)
                
                # SUBSTEP 7c: FORWARD PASS - MAKE PREDICTIONS
                # Feed data through model to get predictions and calculate loss
                # Loss measures how wrong the predictions are compared to correct answers
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                loss = outputs.loss
                
                # SUBSTEP 7d: BACKWARD PASS - LEARN FROM MISTAKES
                # Calculate gradients: how much each model weight contributed to the error
                loss.backward()
                
                # SUBSTEP 7e: GRADIENT CLIPPING
                # Prevent exploding gradients by limiting their magnitude
                # This stabilizes training and prevents wild weight updates
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # SUBSTEP 7f: UPDATE MODEL WEIGHTS
                # Use calculated gradients to improve model weights
                optimizer.step()    # Update weights based on gradients
                scheduler.step()    # Update learning rate schedule
                
                # SUBSTEP 7g: TRACK PROGRESS WITH TIMING
                batch_time = time.time() - batch_start_time
                epoch_loss += loss.item()
                print(f"   Batch {batch_idx + 1}: loss = {loss.item():.4f} ({batch_time:.1f}s)")
                sys.stdout.flush()  # Ensure output appears immediately
                
                # SUBSTEP 7h: MEMORY MANAGEMENT FOR BACKGROUND PROCESSING
                # Clean up GPU memory to prevent accumulation during long training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Allow other processes to run (prevents UI freezing)
                time.sleep(0.01)  # Small delay to keep system responsive
            
            # EPOCH COMPLETION WITH AUTO-SAVE
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"✅ [EPOCH {epoch + 1}] Completed - Loss: {avg_loss:.4f} ({epoch_time:.1f}s)")
            sys.stdout.flush()
            
            # AUTO-SAVE AFTER EACH EPOCH (prevents loss of progress)
            if epoch < epochs - 1:  # Don't double-save on final epoch
                temp_save_path = f"{MODEL_SAVE_PATH}_epoch_{epoch + 1}"
                print(f"💾 [AUTO-SAVE] Saving progress after epoch {epoch + 1}...")
                try:
                    model.save_pretrained(temp_save_path)
                    print(f"✅ [AUTO-SAVE] Progress saved to {temp_save_path}")
                except Exception as save_error:
                    print(f"⚠️ [AUTO-SAVE] Warning: Could not save progress: {save_error}")
                sys.stdout.flush()
            
            # MEMORY CLEANUP BETWEEN EPOCHS
            gc.collect()  # Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # STEP 8: FINAL SAVE OF THE IMPROVED MODEL
        print(f"💾 [FINAL SAVE] Saving completed fine-tuned model...")
        sys.stdout.flush()
        
        model.save_pretrained(MODEL_SAVE_PATH)      # Save model weights and config
        tokenizer.save_pretrained(MODEL_SAVE_PATH)  # Save tokenizer settings
        print(f"🎉 [SUCCESS] Model saved to {MODEL_SAVE_PATH}")
        print(f"🎯 [COMPLETE] Background training finished successfully!")
        sys.stdout.flush()
        return True
        
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print(f"\n⏹️ [INTERRUPTED] Training stopped by user")
        print(f"💾 [RECOVERY] Attempting to save current progress...")
        sys.stdout.flush()
        try:
            model.save_pretrained(f"{MODEL_SAVE_PATH}_interrupted")
            print(f"✅ [RECOVERY] Progress saved to {MODEL_SAVE_PATH}_interrupted")
        except:
            print(f"❌ [RECOVERY] Could not save interrupted progress")
        sys.stdout.flush()
        return False
        
    except Exception as e:
        # Handle any unexpected errors during training
        print(f"\n❌ [ERROR] Training failed with error: {e}")
        print(f"💾 [RECOVERY] Attempting to save any progress...")
        sys.stdout.flush()
        try:
            model.save_pretrained(f"{MODEL_SAVE_PATH}_error_recovery")
            print(f"✅ [RECOVERY] Progress saved to {MODEL_SAVE_PATH}_error_recovery")
        except:
            print(f"❌ [RECOVERY] Could not save error recovery")
        sys.stdout.flush()
        return False
# This function is used to chunk text into manageable pieces for processing
# It ensures that each chunk does not exceed the maximum token limit of the model
def chunk_text(text, max_tokens=4000, stride=500):
    """
    Breaks down large RFP documents into smaller chunks that fit within model token limits.
    This function uses a sliding window approach with configurable stride to ensure
    important information isn't lost at chunk boundaries. Each chunk is tokenized,
    limited to max_tokens, and then decoded back to text for processing.
    Essential for handling long RFP documents that exceed Longformer's 4096 token limit.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"[DEBUG] Total tokens in document: {len(tokens)}")
    chunks = []
    for i in range(0, len(tokens), max_tokens - stride):
        chunk = tokens[i:i + max_tokens]
        print(f"[DEBUG] Chunk {len(chunks)}: Tokens {i} to {i+len(chunk)}")
        chunks.append(tokenizer.decode(chunk))
    return chunks

# This function extracts key features from RFP text for similarity comparison
def extract_rfp_features(text):
    """
    Extracts key RFP-specific features and patterns from document text for similarity analysis.
    This function uses regex patterns to identify important RFP elements like budget information,
    deadlines, project duration, required services, qualifications, and scope of work.
    It combines these extracted features with general context to create a feature-rich
    representation used for comparing documents with historical decisions.
    Returns a concatenated string of relevant features for TF-IDF vectorization.
    """
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
    """
    Loads historical RFP decision data from the CSV database for model training and evaluation.
    This function reads the rfp_db.csv file and filters for entries with actual decisions
    (either 'Approved' or 'Denied'), excluding any pending or incomplete entries.
    Returns a pandas DataFrame containing historical decisions with their associated
    document summaries, which serves as the foundation for model learning and accuracy assessment.
    """
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
    """
    Calculates similarity score between current document and historical RFP decisions.
    This function implements a sophisticated similarity analysis using TF-IDF vectorization
    and cosine similarity to compare the current document against historically approved
    and denied RFPs. It extracts features from all documents, computes similarity scores,
    and returns a probability score indicating how similar the current document is to
    previously approved versus denied documents. Higher scores indicate greater similarity
    to approved documents, while lower scores suggest similarity to denied documents.
    """
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
    """
    Generates enhanced probability predictions for RFP documents using hybrid approach.
    This is the main prediction function that combines fine-tuned Longformer model outputs
    with historical similarity learning. It automatically triggers fine-tuning when sufficient
    new data is available, processes documents in chunks to handle long texts, and weights
    the final prediction based on model confidence and historical pattern matching.
    Returns a probability score (0-1) where higher values indicate greater likelihood of approval.
    """
    global model
    
    print(f"[DEBUG] Starting enhanced prediction for document...")
    
    # Auto-training with progressive thresholds: 5 entries first, then every 10
    if auto_finetune:
        historical_df = load_historical_decisions()
        num_decisions = len(historical_df)
        has_model = os.path.exists(MODEL_SAVE_PATH)
        
        should_train = False
        
        # Get last training count from tracking file
        last_trained_count = get_last_training_count()
        
        # Debug info
        print(f"[DEBUG] Training check: {num_decisions} decisions, last trained at {last_trained_count}")
        
        if not has_model and num_decisions >= 5:
            # First training at 5 entries
            should_train = True
            print(f"🎯 [AUTO-TRAIN] First training trigger at {num_decisions} entries")
        elif has_model and num_decisions >= last_trained_count + 10:
            # Subsequent training every 10 entries
            should_train = True
            print(f"🔄 [AUTO-TRAIN] Progressive training trigger: {num_decisions} total entries ({num_decisions - last_trained_count} since last training)")
        
        if should_train:
            print(f"🤖 [AUTO-TRAIN] Starting automatic fine-tuning...")
            try:
                fine_tune_success = fine_tune_model(epochs=2, batch_size=1)  # 2 epochs for better learning
                if fine_tune_success:
                    # Reload model and save training count
                    model = LongformerForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
                    save_training_count(num_decisions)  # Track when we last trained
                    print("✅ [AUTO-TRAIN] Model updated successfully")
                else:
                    print("❌ [AUTO-TRAIN] Training failed, continuing with existing model")
            except Exception as e:
                print(f"⚠️ [AUTO-TRAIN] Training error: {e}")
                print("Continuing with existing model...")
    
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
    """
    SIMPLIFIED manual fine-tuning trigger with clear feedback.
    This provides an easy way to improve the model when you have new decisions.
    """
    print("🎯 [MANUAL TRAINING] Initiating model improvement...")
    
    # Quick data check
    historical_df = load_historical_decisions()
    approved_count = len(historical_df[historical_df['decision'] == 'Approved']) if not historical_df.empty else 0
    denied_count = len(historical_df[historical_df['decision'] == 'Denied']) if not historical_df.empty else 0
    total_decisions = len(historical_df)
    
    print(f"📊 [DATA CHECK] Found {total_decisions} decisions ({approved_count} approved, {denied_count} denied)")
    
    if total_decisions < 2:
        print(f"⚠️ [INSUFFICIENT DATA] Need at least 2 decisions to train. Currently have {total_decisions}.")
        print("💡 [TIP] Make some approve/deny decisions first, then try training again.")
        return False
    
    if approved_count == 0 or denied_count == 0:
        print(f"⚠️ [UNBALANCED DATA] Need both approved AND denied examples.")
        print(f"   Current: {approved_count} approved, {denied_count} denied")
        print("💡 [TIP] Make decisions in both categories for better training.")
        return False
    
    print(f"✅ [READY] Sufficient data for training!")
    print(f"🚀 [STARTING] Beginning simplified fine-tuning process...")
    
    success = fine_tune_model(epochs=2, batch_size=1)  # Simplified parameters
    
    if success:
        save_training_count(total_decisions)  # Track manual training
        print(f"🎉 [SUCCESS] Model training completed!")
        print(f"💡 [NEXT STEPS] Your model is now personalized to your decisions.")
        print(f"🔄 [SUGGESTION] Consider running 'Rerun Model' to update existing predictions.")
    else:
        print(f"❌ [FAILED] Training didn't complete successfully.")
        print(f"🔍 [DEBUG] Check the logs above for specific errors.")
    
    return success

def evaluate_model_accuracy(threshold=0.5, test_size=0.3, verbose=True):
    """
    Comprehensive accuracy evaluation system for the RFP classification model.
    This function performs rigorous cross-validation testing using historical decisions
    to assess model performance. It splits data into training/testing sets, generates
    predictions, and calculates detailed metrics including overall accuracy, precision,
    recall, F1-scores, confusion matrix, and probability calibration metrics.
    Provides both quantitative performance measures and detailed prediction breakdowns
    to help understand model strengths and weaknesses.
    
    Args:
        threshold: Probability threshold for classification (default: 0.5)
        test_size: Proportion of data to use for testing (default: 0.3)
        verbose: Whether to print detailed results (default: True)
    
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    historical_df = load_historical_decisions()
    
    if len(historical_df) < 4:
        result = {
            'error': f'Insufficient data for evaluation. Need at least 4 decisions, but only have {len(historical_df)}',
            'data_count': len(historical_df)
        }
        if verbose:
            print(f"[ERROR] {result['error']}")
        return result
    
    if verbose:
        print(f"[INFO] Evaluating model accuracy with {len(historical_df)} historical decisions")
        print(f"[INFO] Using threshold: {threshold}, test size: {test_size}")
    
    # Prepare data
    texts = []
    true_labels = []
    
    for _, row in historical_df.iterrows():
        summary = row.get('summary', '')
        if summary and len(summary.strip()) > 20:
            texts.append(summary)
            true_labels.append(1 if row['decision'] == 'Approved' else 0)
    
    if len(texts) < 4:
        result = {
            'error': f'Insufficient valid text data. Need at least 4 summaries, but only have {len(texts)}',
            'valid_summaries': len(texts)
        }
        if verbose:
            print(f"[ERROR] {result['error']}")
        return result
    
    # Split data for evaluation
    # Check if we can use stratification (need at least 2 examples of each class)
    label_counts = dict(zip(*np.unique(true_labels, return_counts=True)))
    can_stratify = len(set(true_labels)) > 1 and all(count >= 2 for count in label_counts.values())
    
    if can_stratify:
        if verbose:
            print("[DEBUG] Using stratified split for evaluation")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, true_labels, test_size=test_size, random_state=42, stratify=true_labels
        )
    else:
        if verbose:
            print("[DEBUG] Using random split for evaluation (insufficient data for stratification)")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, true_labels, test_size=test_size, random_state=42
        )
    
    if verbose:
        print(f"[INFO] Training set: {len(train_texts)} samples")
        print(f"[INFO] Test set: {len(test_texts)} samples")
    
    # Get predictions on test set
    predictions = []
    probabilities = []
    
    for text in test_texts:
        prob = predict_document_probability(text, auto_finetune=False)  # Don't auto-finetune during evaluation
        probabilities.append(prob)
        predictions.append(1 if prob > threshold else 0)
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    total = len(test_labels)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate precision, recall, F1 for each class
    tp_approved = sum(1 for pred, true in zip(predictions, test_labels) if pred == 1 and true == 1)
    fp_approved = sum(1 for pred, true in zip(predictions, test_labels) if pred == 1 and true == 0)
    fn_approved = sum(1 for pred, true in zip(predictions, test_labels) if pred == 0 and true == 1)
    tn_approved = sum(1 for pred, true in zip(predictions, test_labels) if pred == 0 and true == 0)
    
    precision_approved = tp_approved / (tp_approved + fp_approved) if (tp_approved + fp_approved) > 0 else 0
    recall_approved = tp_approved / (tp_approved + fn_approved) if (tp_approved + fn_approved) > 0 else 0
    f1_approved = 2 * (precision_approved * recall_approved) / (precision_approved + recall_approved) if (precision_approved + recall_approved) > 0 else 0
    
    precision_denied = tn_approved / (tn_approved + fn_approved) if (tn_approved + fn_approved) > 0 else 0
    recall_denied = tn_approved / (tn_approved + fp_approved) if (tn_approved + fp_approved) > 0 else 0
    f1_denied = 2 * (precision_denied * recall_denied) / (precision_denied + recall_denied) if (precision_denied + recall_denied) > 0 else 0
    
    # Calculate mean absolute error for probability predictions
    mae = sum(abs(prob - true) for prob, true in zip(probabilities, test_labels)) / len(probabilities)
    
    result = {
        'overall_accuracy': accuracy,
        'correct_predictions': correct,
        'total_predictions': total,
        'threshold_used': threshold,
        'test_size': test_size,
        'confusion_matrix': {
            'true_positive_approved': tp_approved,
            'false_positive_approved': fp_approved,
            'false_negative_approved': fn_approved,
            'true_negative_approved': tn_approved
        },
        'approved_metrics': {
            'precision': precision_approved,
            'recall': recall_approved,
            'f1_score': f1_approved
        },
        'denied_metrics': {
            'precision': precision_denied,
            'recall': recall_denied,
            'f1_score': f1_denied
        },
        'probability_mae': mae,
        'predictions_vs_actual': list(zip(probabilities, predictions, test_labels))
    }
    
    if verbose:
        print_accuracy_report(result)
    
    return result

def print_accuracy_report(metrics):
    """
    Formats and displays a comprehensive accuracy evaluation report to the console.
    This function takes the metrics dictionary from evaluate_model_accuracy and
    presents the results in a well-formatted, easy-to-read report including
    overall accuracy, confusion matrix, per-class performance metrics, and
    detailed prediction breakdowns. Provides visual indicators for correct/incorrect
    predictions and organizes information for quick assessment of model performance.
    """
    print("\n" + "="*60)
    print("           RFP MODEL ACCURACY EVALUATION REPORT")
    print("="*60)
    
    if 'error' in metrics:
        print(f"ERROR: {metrics['error']}")
        return
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_predictions']}")
    print(f"Threshold Used: {metrics['threshold_used']}")
    print(f"Probability MAE: {metrics['probability_mae']:.4f}")
    
    print("\nCONFUSION MATRIX:")
    print("                    Predicted")
    print("                Approved  Denied")
    print(f"Actual Approved    {metrics['confusion_matrix']['true_positive_approved']:4d}      {metrics['confusion_matrix']['false_negative_approved']:4d}")
    print(f"       Denied      {metrics['confusion_matrix']['false_positive_approved']:4d}      {metrics['confusion_matrix']['true_negative_approved']:4d}")
    
    print("\nPER-CLASS METRICS:")
    print("Class: APPROVED")
    print(f"  Precision: {metrics['approved_metrics']['precision']:.2%}")
    print(f"  Recall:    {metrics['approved_metrics']['recall']:.2%}")
    print(f"  F1-Score:  {metrics['approved_metrics']['f1_score']:.2%}")
    
    print("Class: DENIED")
    print(f"  Precision: {metrics['denied_metrics']['precision']:.2%}")
    print(f"  Recall:    {metrics['denied_metrics']['recall']:.2%}")
    print(f"  F1-Score:  {metrics['denied_metrics']['f1_score']:.2%}")
    
    print("\nDETAILED PREDICTIONS:")
    print("Probability | Predicted | Actual | Correct?")
    print("-" * 42)
    for prob, pred, actual in metrics['predictions_vs_actual']:
        pred_label = "Approved" if pred == 1 else "Denied"
        actual_label = "Approved" if actual == 1 else "Denied"
        correct = "✓" if pred == actual else "✗"
        print(f"   {prob:.3f}    |  {pred_label:8s} | {actual_label:7s} |   {correct}")
    
    print("="*60)

def run_accuracy_evaluation():
    """
    Executes a comprehensive accuracy evaluation workflow with threshold optimization.
    This function orchestrates the complete evaluation process by first displaying
    current model information, then testing multiple probability thresholds to find
    the optimal one, and finally running a detailed evaluation with the best threshold.
    It provides a complete assessment of model performance including threshold sensitivity
    analysis and detailed metrics reporting. This is the main entry point for
    comprehensive model evaluation from the command line interface.
    """
    print("Starting RFP Model Accuracy Evaluation...")
    
    # Check model info first
    model_info = get_model_info()
    print(f"\nModel Information:")
    print(f"  Fine-tuned model available: {model_info['has_fine_tuned_model']}")
    print(f"  Historical decisions: {model_info['historical_decisions']}")
    print(f"  Approved: {model_info['approved_count']}, Denied: {model_info['denied_count']}")
    print(f"  Model path: {model_info['model_path']}")
    
    # Run evaluation with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_accuracy = 0
    best_threshold = 0.5
    
    print(f"\nTesting different probability thresholds...")
    print("Threshold | Accuracy")
    print("-" * 20)
    
    for threshold in thresholds:
        result = evaluate_model_accuracy(threshold=threshold, verbose=False)
        if 'error' not in result:
            accuracy = result['overall_accuracy']
            print(f"   {threshold:.1f}    | {accuracy:.2%}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        else:
            print(f"   {threshold:.1f}    | Error: {result['error']}")
            return
    
    print(f"\nBest threshold: {best_threshold} (Accuracy: {best_accuracy:.2%})")
    
    # Run detailed evaluation with best threshold
    print(f"\nRunning detailed evaluation with threshold {best_threshold}...")
    evaluate_model_accuracy(threshold=best_threshold, verbose=True)

def get_model_info():
    """
    Retrieves and returns current model state and historical data statistics.
    This function provides a comprehensive overview of the model's current configuration
    including whether a fine-tuned model exists, the amount of historical training data,
    the distribution of approved vs denied decisions, and the model path being used.
    Essential for understanding model readiness and data availability before running
    evaluations or making predictions. Used by evaluation tools and diagnostic functions.
    """
    print("[DEBUG] Getting model info...")
    historical_df = load_historical_decisions()
    has_finetuned = os.path.exists(MODEL_SAVE_PATH)
    
    approved_count = len(historical_df[historical_df['decision'] == 'Approved']) if not historical_df.empty else 0
    denied_count = len(historical_df[historical_df['decision'] == 'Denied']) if not historical_df.empty else 0
    
    info = {
        'has_fine_tuned_model': has_finetuned,
        'historical_decisions': len(historical_df),
        'approved_count': approved_count,
        'denied_count': denied_count,
        'model_path': MODEL_SAVE_PATH if has_finetuned else MODEL_NAME
    }
    
    print(f"[DEBUG] Model info: {info}")
    return info


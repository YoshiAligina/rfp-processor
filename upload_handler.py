# Upload Handler - Handles file upload and processing logic for Flask

"""
File Upload and Processing Handler for RFP Analysis System

This module manages the complete file upload workflow including metadata collection,
document processing, ML prediction generation, and database storage. It supports
both individual document processing and project-grouped submissions where multiple
files belong to a single RFP. The module integrates text extraction, OCR processing,
and intelligent summarization to create comprehensive RFP entries.

Key features:
- Flexible upload modes (individual documents vs. grouped projects)
- Multi-format document processing (PDF, DOCX, Excel)
- Real-time ML probability prediction
- Intelligent document summarization
- Database integration with duplicate detection
"""

import os
import re
from ocr_utils import is_scanned, extract_text
from model_utils import predict_document_probability
from data_utils import add_entry_to_db, load_db
from document_utils import extract_text_from_file, get_file_type

PDF_FOLDER = "documents"

def generate_document_summary(text, max_length=200):
    """
    Generate a concise summary of the document text.
    
    Args:
        text (str): The full document text
        max_length (int): Maximum length of the summary
        
    Returns:
        str: A concise summary of the document
    """
    if not text or len(text.strip()) < 50:
        return "Document content too short for meaningful summary."
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Extract key sentences (simple approach)
    sentences = text.split('.')
    important_sentences = []
    
    # Look for sentences with key RFP terms
    key_terms = ['rfp', 'request for proposal', 'bid', 'contract', 'vendor', 'supplier', 
                 'deadline', 'requirements', 'scope', 'budget', 'proposal', 'submission']
    
    for sentence in sentences[:10]:  # Only check first 10 sentences
        sentence = sentence.strip()
        if len(sentence) > 20:  # Ignore very short sentences
            if any(term in sentence.lower() for term in key_terms):
                important_sentences.append(sentence)
        if len(important_sentences) >= 3:  # Limit to 3 key sentences
            break
    
    if important_sentences:
        summary = '. '.join(important_sentences) + '.'
    else:
        # Fallback: use first few sentences
        summary = '. '.join(sentences[:2]).strip() + '.'
    
    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
    
    return summary

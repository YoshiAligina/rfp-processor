# UI Components - Reusable UI elements and helper functions

import streamlit as st
import re

def generate_document_summary(text, max_sentences=3):
    """Generate a summary of document text based on key sentences"""
    if not text or len(text.strip()) < 100:
        return "Document too short to summarize effectively."
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= max_sentences:
        return ' '.join(sentences[:max_sentences]) + '.'
    
    # Simple scoring based on length and position (earlier sentences often more important)
    scored_sentences = []
    for i, sentence in enumerate(sentences[:10]):  # Only consider first 10 sentences
        # Score based on length (prefer medium-length sentences) and position (prefer earlier)
        length_score = min(len(sentence) / 100, 1.0)  # Normalize to 0-1
        position_score = (10 - i) / 10  # Earlier sentences get higher score
        
        # Look for key RFP-related terms
        key_terms = ['request', 'proposal', 'requirements', 'services', 'project', 'deadline', 'budget', 'scope']
        term_score = sum(1 for term in key_terms if term.lower() in sentence.lower()) / len(key_terms)
        
        total_score = length_score * 0.3 + position_score * 0.4 + term_score * 0.3
        scored_sentences.append((sentence, total_score))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    summary_sentences = [s[0] for s in scored_sentences[:max_sentences]]
    
    original_order = []
    for sentence in sentences:
        if sentence in summary_sentences:
            original_order.append(sentence)
            if len(original_order) == max_sentences:
                break
    
    return ' '.join(original_order) + '.'

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>RFP Analyzer & Processor</h1>
    </div>
    """, unsafe_allow_html=True)

def render_empty_state():
    """Render the empty state when no documents are uploaded"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #ffffff; border-radius: 10px; border: 2px dashed #B9C930; box-shadow: 0 2px 4px rgba(185, 201, 48, 0.1);">
        <h3 style="color: #4A7637;">No documents uploaded yet</h3>
        <p style="color: #4A7637;">Upload your first RFP document (PDF, Word, or Excel) in the "Upload & Process" tab to get started!</p>
    </div>
    """, unsafe_allow_html=True)

# UI Components - Reusable UI elements and helper functions

"""
User Interface Components for RFP Processing Application

This module provides reusable UI components and utility functions for the Streamlit-based
RFP analysis interface. It contains specialized functions for document summarization,
header rendering, and empty state displays. The components are designed to maintain
consistent styling and user experience across the application while providing
intelligent content processing capabilities.

Key components:
- Intelligent document summarization with RFP-specific scoring
- Consistent application header rendering
- Empty state displays for better user guidance
- Reusable UI patterns for maintainable interface code
"""

import streamlit as st
import re

def generate_document_summary(text, max_sentences=3):
    """
    Generates intelligent document summaries optimized for RFP content analysis.
    This function creates concise, meaningful summaries by scoring sentences based
    on multiple factors: length optimization, position weighting, and RFP-specific
    terminology detection. It identifies the most relevant sentences from document
    content and presents them in their original order for better readability.
    Essential for providing users with quick document overviews in the database
    interface without requiring full document review.
    
    Args:
        text (str): Full document text content to summarize
        max_sentences (int): Maximum number of sentences to include in summary
        
    Returns:
        str: Intelligent summary preserving original sentence order and key information
    """
    if not text or len(text.strip()) < 100:
        return "Document too short to summarize effectively."
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= max_sentences:
        return ' '.join(sentences[:max_sentences]) + '.'
    
    scored_sentences = []
    for i, sentence in enumerate(sentences[:40]):  
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
    """
    Renders the main application header with consistent branding and styling.
    This function creates the primary application header using custom HTML and CSS
    styling to maintain visual consistency across all application pages. It provides
    professional branding for the RFP analysis tool and establishes the visual
    hierarchy for the user interface. The header serves as a navigation anchor
    and brand identity element throughout the application experience.
    """
    st.markdown("""
    <div class="main-header">
        <h1>RFP Model</h1>
    </div>
    """, unsafe_allow_html=True)

def render_empty_state():
    """
    Displays user-friendly empty state interface when no documents are present.
    This function creates an engaging, informative display for new users or when
    the database is empty, providing clear guidance on how to get started with
    the application. It uses custom styling to create an attractive, professional
    appearance that encourages user engagement and clearly communicates next steps.
    Essential for user onboarding and maintaining good user experience during
    initial application setup or after database resets.
    """
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #ffffff; border-radius: 10px; border: 2px dashed #B9C930; box-shadow: 0 2px 4px rgba(185, 201, 48, 0.1);">
        <h3 style="color: #4A7637;">No documents uploaded yet</h3>
        <p style="color: #4A7637;">Upload your first RFP document (PDF, Word, or Excel) in the "Upload & Process" tab to get started!</p>
    </div>
    """, unsafe_allow_html=True)

# RFP Analyzer & Processor - Main Application Entry Point
# This is the main entry point for the Flask-based RFP processing application

"""
RFP (Request for Proposal) Analyzer & Processor - Application Entry Point

This module serves as the primary entry point for the RFP processing application.
It starts the Flask web application for production use. The application provides
AI-powered analysis of RFP documents, including:
- Document text extraction and OCR processing
- ML-based approval/denial probability scoring
- Historical decision tracking and learning
- Professional web interface for document management
- Comprehensive accuracy evaluation tools
"""

from web_app import app

if __name__ == "__main__":
    print("="*60)
    print("   RFP ANALYZER - STARTING APPLICATION")
    print("="*60)
    print("🚀 Starting Flask web application...")
    print("🌐 Navigate to http://localhost:5000 in your browser")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# RFP Analyzer & Processor - Main Application Entry Point
# This is the main entry point - the actual logic is now modularized in separate files

"""
RFP (Request for Proposal) Analyzer & Processor - Application Entry Point

This module serves as the primary entry point for the RFP processing application.
It delegates the main application logic to the main_app module, following a clean
separation of concerns where this file handles application startup and main_app
contains the core application implementation. This design pattern allows for
better code organization, easier testing, and cleaner module imports.

The application provides AI-powered analysis of RFP documents, including:
- Document text extraction and OCR processing
- ML-based approval/denial probability scoring
- Historical decision tracking and learning
- Interactive web interface for document management
- Comprehensive accuracy evaluation tools
"""

from main_app import main

if __name__ == "__main__":
    main()

"""
Optical Character Recognition (OCR) Utilities for RFP Document Processing

This module provides advanced OCR capabilities for processing scanned PDF documents
and image-based content that cannot be extracted through standard text parsing.
It integrates EasyOCR for high-quality text recognition and provides intelligent
detection of scanned vs. text-based documents to optimize processing workflows.

Key features:
- Automatic scanned document detection
- High-quality OCR using EasyOCR with English language support
- Dual extraction methods (standard PDF text extraction + OCR fallback)
- Image preprocessing for optimal OCR results
- Integration with PyMuPDF for document handling
"""

import fitz  # PyMuPDF
import PyPDF2
import easyocr

# Initialize EasyOCR reader with English language support, CPU mode for compatibility
reader = easyocr.Reader(['en'], gpu=False)

def is_scanned(pdf_path):
    """
    Intelligently detects whether a PDF document contains scanned images or extractable text.
    This function analyzes PDF pages to determine if they contain selectable text or are
    image-based scanned documents requiring OCR processing. It checks each page for
    extractable text content and returns True if no text is found, indicating the
    document is likely scanned. Essential for routing documents to appropriate
    processing methods and avoiding unnecessary OCR operations on text-based documents.
    
    Args:
        pdf_path (str): Full path to the PDF document to analyze
        
    Returns:
        bool: True if document appears to be scanned (no extractable text), False otherwise
    """
    doc = fitz.open(pdf_path)
    for page in doc:
        if page.get_text():
            return False
    return True

def extract_text(pdf_path, use_easyocr=False):
    """
    Extracts text from PDF documents using either standard parsing or advanced OCR.
    This function provides dual extraction modes: standard text extraction for
    text-based PDFs using PyPDF2, and OCR-based extraction for scanned documents
    using EasyOCR. The OCR mode converts PDF pages to images and applies character
    recognition to extract text from visual content. Handles both processing
    workflows seamlessly based on document characteristics and user preferences.
    
    Args:
        pdf_path (str): Full path to the PDF document to process
        use_easyocr (bool): Force OCR usage even for text-based PDFs
        
    Returns:
        str: Extracted text content from the PDF document
    """
    if use_easyocr:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img_bytes = pix.tobytes("png")
            result = reader.readtext(img_bytes, detail=0)
            text += " ".join(result) + "\n"
        return text
    else:
        with open(pdf_path, "rb") as file:
            reader_pdf = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() or "" for page in reader_pdf.pages])
        return text

def convert_scanned(pdf_path, output_path="processed.pdf"):
    """
    Legacy function for scanned document processing, now delegates to OCR text extraction.
    This function maintains backward compatibility for older code that expected
    document conversion functionality. It now simply calls the OCR-based text
    extraction method and returns the extracted text rather than creating a new
    processed document file. Kept for API compatibility while leveraging the
    improved OCR extraction capabilities of the extract_text function.
    
    Args:
        pdf_path (str): Path to the scanned PDF document
        output_path (str): Legacy parameter, no longer used but kept for compatibility
        
    Returns:
        str: OCR-extracted text content from the scanned document
    """
    return extract_text(pdf_path, use_easyocr=True)

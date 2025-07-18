import fitz  # PyMuPDF
import ocrmypdf
import PyPDF2
import os

def is_scanned(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        if page.get_text():
            return False
    return True

def convert_scanned(pdf_path, output_path="processed.pdf"):
    ocrmypdf.ocr(pdf_path, output_path)
    return output_path

def extract_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
    return text

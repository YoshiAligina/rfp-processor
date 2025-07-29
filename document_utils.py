import os
from docx import Document
import pandas as pd
import openpyxl
from ocr_utils import is_scanned, extract_text as extract_pdf_text

def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    try:
        doc = Document(file_path)
        text_content = []
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text.strip())
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_content.append(" | ".join(row_text))
        
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_excel(file_path):
    """Extract text from Excel files (xlsx, xls)"""
    try:
        # Try reading with openpyxl first (for .xlsx)
        try:
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            text_content = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_content.append(f"Sheet: {sheet_name}")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = []
                    for cell in row:
                        if cell is not None and str(cell).strip():
                            row_text.append(str(cell).strip())
                    if row_text:
                        text_content.append(" | ".join(row_text))
            
            return "\n".join(text_content)
            
        except Exception:
            # Fallback to pandas for both .xlsx and .xls
            excel_file = pd.ExcelFile(file_path)
            text_content = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text_content.append(f"Sheet: {sheet_name}")
                
                # Convert DataFrame to text
                for _, row in df.iterrows():
                    row_text = []
                    for value in row:
                        if pd.notna(value) and str(value).strip():
                            row_text.append(str(value).strip())
                    if row_text:
                        text_content.append(" | ".join(row_text))
            
            return "\n".join(text_content)
            
    except Exception as e:
        print(f"Error extracting text from Excel: {e}")
        return ""

def get_file_type(filename):
    """Get the file type based on extension"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext == '.docx':
        return 'docx'
    elif ext in ['.xlsx', '.xls']:
        return 'excel'
    else:
        return 'unknown'

def extract_text_from_file(file_path, use_ocr=False):
    """Extract text from any supported file type"""
    file_type = get_file_type(file_path)
    
    if file_type == 'pdf':
        if is_scanned(file_path):
            return extract_pdf_text(file_path, use_easyocr=True)
        else:
            return extract_pdf_text(file_path)
    elif file_type == 'docx':
        return extract_text_from_docx(file_path)
    elif file_type == 'excel':
        return extract_text_from_excel(file_path)
    else:
        return ""

def get_appropriate_mime_type(filename):
    """Get appropriate MIME type for file download"""
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel'
    }
    return mime_types.get(ext, 'application/octet-stream')

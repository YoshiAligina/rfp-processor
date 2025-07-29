import fitz  # PyMuPDF
import PyPDF2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def is_scanned(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        if page.get_text():
            return False
    return True

def extract_text(pdf_path, use_easyocr=False):
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
    return extract_text(pdf_path, use_easyocr=True)

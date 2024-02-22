import fitz  # PyMuPDF



def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    
    full_text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)

        text = page.get_text()

        full_text += text
    
    document.close()
    
    return full_text
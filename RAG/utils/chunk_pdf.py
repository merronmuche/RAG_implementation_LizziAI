

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


def chunk_text(text, words_per_chunk=500, overlap_size=20):
    # Split the text into words
    words = text.split()

    chunks = []
    i = 0

    while i < len(words):
        # Calculate the end index for the current chunk
        end = i + words_per_chunk

        # Create a chunk as a string of words
        chunk = ' '.join(words[i:end])
        chunks.append(chunk)

        # Calculate the start index for the next chunk, considering the overlap
        i = end - overlap_size if end - overlap_size > i else end

    return chunks


if __name__ == "__main__":
    pdf_path = "data/week6_data.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    # print(extracted_text)

    chunks = chunk_text(extracted_text, 150, 5)
    print(len(chunks))

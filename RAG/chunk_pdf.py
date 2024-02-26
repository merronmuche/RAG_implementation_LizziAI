
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    base_docs = [p.text for p in doc.paragraphs]
    return base_docs

file_path = "/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Raptor Contract.docx"
doc_text = read_docx(file_path)
print(doc_text)

def chunk_text(docs):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500)
    return [chunk for doc in docs for chunk in chunks.split_text(doc)]


# Split the document text into chunks
doc_chunks = chunk_text(doc_text)
print(len(doc_chunks))




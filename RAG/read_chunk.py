
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    base_doc = ''
    for p in doc.paragraphs: 
        base_doc += p.text 
    return base_doc

def chunk_text(doc):
    chunks = RecursiveCharacterTextSplitter(chunk_size=2000)
    x = chunks.split_text(doc)
    return x


if  __name__=='__main__':
    file_path = "/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Raptor Contract.docx"
    doc_text = read_docx(file_path)
    print(doc_text)
   







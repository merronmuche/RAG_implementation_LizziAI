
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

from . chunk_pdf import extract_text_from_pdf, chunk_text

load_dotenv()

def embed_text(chunks):
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    embeds = embed_model.embed_documents(chunks)

    return embeds


if __name__ == "__main__":

    pdf_path = "data/week6_data.pdf"
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, 150, 5)

    embeds = embed_text(chunks)
    print(embeds)



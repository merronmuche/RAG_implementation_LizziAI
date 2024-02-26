
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv


from . chunk_pdf import read_docx, chunk_text

load_dotenv()

def embed_text(chunks):
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    embeds = embed_model.embed_documents(chunks)

    return embeds

if __name__ == "__main__":

    pdf_path = "/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Robinson Advisory.docx"
    
    text = read_docx(docx.path)
    chunks = chunk_text(text, 150, 5)

    embeds = embed_text(chunks)
    print(embeds)



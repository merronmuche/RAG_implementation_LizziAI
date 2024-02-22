from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from chunk_document import extract_text_from_pdf

# Load your PDF document
path = 'datalizzy/Raptor Contract.docx'
path1 = 'datalizzy/Robinson Advisory.docx'
docs1 = extract_text_from_pdf(path)
docs2 = extract_text_from_pdf(path1)
docs = docs1 + docs2

# Use the parent splitter to split the documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
split_docs = parent_splitter.split_documents(docs)

# Initialize the embeddings from langchain_openai
embedding = OpenAIEmbeddings(chunk_size=1)

# Initialize the vectorstore and store
vectorstore = Chroma(collection_name="full_documents", embedding_function=embedding)
store = InMemoryStore()

# Initialize the retriever with parent and child chunking
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20),
    parent_splitter=parent_splitter
)

# Add the split documents to the retriever
retriever.add_documents(split_docs, ids=None)

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from chunk_document import extract_text_from_pdf

from langchain.document_loaders import PyMuPDFLoader, LarkSuiteDocLoader, UnstructuredWordDocumentLoader

# path = '/home/meron/tutorial/Advanced-RAG/Raptor_Contract.pdf'
path = '/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Raptor Contract.docx'
path1 = '/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Robinson Advisory.docx'
loader = UnstructuredWordDocumentLoader(path)
doc1= loader.load()
loader = UnstructuredWordDocumentLoader(path1)
doc2= loader.load()

docs = doc1 + doc2
text = docs[5].page_content
print(text)




from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=embedding
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
# Add the split documents to the retriever
retriever.add_documents(docs, ids=None)
vectorstore.similarity_search("How much is the escrow amount?")
retriever.get_relevant_documents("How much is the escrow amount?")
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOpenAI(
        temperature=0,
        max_tokens=800,
        model_kwargs={"top_p": 0, "frequency_penalty": 0, "presence_penalty": 0},
    )


retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

response = llm()

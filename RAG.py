from lanchain_community.document_loaders import TextLoader
from langchain.textsplitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_base=os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version=os.getenv('AZURE_OPENAI_API_VERSION')

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

loader =TextLoader("movies.txt")
documents =loader.load()


text_splitter =RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs=text_splitter.split_documents(documents)
print("Number of split documents:",len(split_docs))

vector_db =FAISS.from_document(split_docs, embedding_model)
retriever= vector_db.as_retriever()

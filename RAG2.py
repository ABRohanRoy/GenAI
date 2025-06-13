import os
import numpy as np
from dotenv import load_dotenv
import faiss
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("Embedding_AZURE_OPENAI_API_VERSION"),
    chunk_size=10,
)

def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    return ""

def store_vector(text):
    vector = np.array(embedding_model.embed_documents([text])).astype("float32")
    dimension = vector.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vector)
    faiss.write_index(index, "faiss_vector.index")
    print("Vector stored successfully")
    return vector

def search(query):
    index = faiss.read_index("faiss_vector.index")
    query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")
    distances, indices = index.search(query_vector, 1)
    return distances, indices

# Run example
file_path = "movies.txt"
text = read_file(file_path)
if text.strip() != "":
    vector = store_vector(text)

    query = "What is the best action movie?"
    distance, indices = search(query)
    print("Distance:", distance)
    print("Indices:", indices)
else:
    print("Text file is empty or missing.")

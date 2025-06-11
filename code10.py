from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

embedding_model= SentenceTransformer("all-MiniLM-L6-v2")

str="Hello, Marwadi university"
embeddings=embedding_model.encode(str,convert_to_tensor=True)

print(embeddings)

from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

embedding_model= SentenceTransformer("BAAI/bge-small-en-v1.5")

str="Hello, Marwadi university"
embeddings=embedding_model.encode(str,convert_to_tensor=True)

print(embeddings)

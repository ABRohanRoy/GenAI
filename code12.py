from langchain_community.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("Embedding_AZURE_OPENAI_API_VERSION"), 
    chunk_size=10,
)

text = "I love machine learning because I find it intriguing."
embedding = embedding_model.embed_query(text)

print(f"Embedding vector length: {len(embedding)}")
print(embedding)

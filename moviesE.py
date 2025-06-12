from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
load_dotenv()
import numpy as np
import pandas as pd

embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("Embedding_AZURE_OPENAI_API_VERSION"), 
    chunk_size=10,
)

movies = [
    'War of the worlds: War, fiction, action',
    'Inception: Sci-fi, thriller, mystery',
    'Gladiator: Action, drama, historical',
    'The Matrix: Sci-fi, action, dystopian',
    'Interstellar: Sci-fi, drama, adventure',
    'Mad Max: Fury Road: Action, dystopian, adventure',
    '1917: War, drama, history',
    'Edge of Tomorrow: Sci-fi, action, thriller',
    'Dunkirk: War, historical, thriller',
    'Avatar: Sci-fi, action, fantasy',
]

with open("movies.txt", "r") as file:
    movies = [line.strip() for line in file.readlines()]

user_input=input("Enter the type of movies you want to see: ")
user_query=user_input.lower()

query_vector = embedding_model.embed_query(user_input)
doc_vectors = embedding_model.embed_documents(movies)

similarities=cosine_similarity([query_vector], doc_vectors)[0]

df=pd.DataFrame({
    'Document':movies,
    'Similarity':similarities
})

ranked_df =df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
print(ranked_df)

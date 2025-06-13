import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

metadata={
    "topgear.txt": { "tags":["Cars","Bikes","racing","Drivers"]},
    "gaming.txt": { "tags":["Mobile-gaming","PC-gaming","Controller","Playstation"]},
    "workouts.txt":{"tags":["gym", "Calisthenetics","Athletic","Fitness"]}
}
query="What does a tractor do?"

docs_as_text = {filename: " ".join(data["tags"]) for filename, data in metadata.items()}

doc_embeddings = model.encode(list(docs_as_text.values()))
query_embedding =model.encode([query])
similarities=cosine_similarity(query_embedding, doc_embeddings)[0]
sorted_indices=np.argsort(similarities)[::-1]

print("\nTop matches for the query:")
for idx in sorted_indices:
    filename=list(docs_as_text.keys())[idx]
    score=similarities[idx]
    print(f"{filename} --> Similarity Score: {score:.4f}")

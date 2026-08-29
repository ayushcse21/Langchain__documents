from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os

os.environ["HF_HOME"] = r"D:\h_face"

embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

query = "Delhi is the capital of India"

docs = [
    "Delhi is the capital of India",
    "Kolkata is the capital of WestBengal",
    "Paris is the capital of France"
]

query = "which is the capital of Franced"

doc_embed = embeddings.embed_documents(docs)

query_embed = embeddings.embed_query(query)

scores = cosine_similarity([query_embed],doc_embed)

index , score = sorted(list(enumerate(scores[0])),key = lambda x:x[1] )[-1]

print(query)
print(docs[index])
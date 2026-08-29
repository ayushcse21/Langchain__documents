from langchain_huggingface import HuggingFaceEmbeddings

import os

os.environ["HF_HOME"] = r"D:\h_face"

embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

query = "Delhi is the capital of India"

docs = [
    "Delhi is the capital of India",
    "Kolkata is the capital of WestBengal",
    "Paris is the capital of France"
]

vector = embeddings.embed_documents(docs)

print(str(vector))
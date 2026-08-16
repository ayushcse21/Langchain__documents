from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

import os


load_dotenv(dotenv_path="C:/Users/ayush/OneDrive/Desktop/langchain_models/.env")

print("API KEY:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))


llm = HuggingFaceEndpoint(

    repo_id= "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

)

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is the capital of India")

print(result.content)

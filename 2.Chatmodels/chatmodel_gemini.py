from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import os


load_dotenv(dotenv_path="E:\\new_langchain_models\.env")

print("API KEY:", os.getenv("GOOGLE_API_KEY"))
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",max_output_tokens=50)

result = model.invoke("what is the capital of india")

print(result)
print("Running...")
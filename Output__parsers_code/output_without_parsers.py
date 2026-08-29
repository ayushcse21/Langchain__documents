from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-Next",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = 'Write a 5 line summary on  {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'Black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result.content})

result = model.invoke(prompt2)

print(result.content)



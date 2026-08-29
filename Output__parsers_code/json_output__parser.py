from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-Next",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='give me 5 facts about black hole {structured_format}',
    input_variables=[],
    partial_variables={'structured_format' : parser.get_format_instructions()}
)

#prompt = template.format()

#print(prompt)

#result = model.invoke(prompt)

#final_result = parser.parse(result.content)

chain = template |model |parser

final_result = chain.invoke({})

print(final_result)
print(type(final_result))
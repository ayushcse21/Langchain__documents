from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-Next",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


)
model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name : str = Field(description='name of the person')
    age : int = Field(gt=18,description='age of the person')
    city : str = Field(description='name of the city the person belongs to')


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template='give me the name age and the city pf {topic} \n {structured_instructions}',
    input_variables=['topic'],
    partial_variables={'structured_instructions':parser.get_format_instructions()}
)

prompt = template.invoke({'topic' : 'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
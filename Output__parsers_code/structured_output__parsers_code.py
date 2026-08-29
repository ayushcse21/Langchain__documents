from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-Next",
    task = "text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


)


model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name = 'fact1',description ='fact1 about the topic'),
    ResponseSchema(name = 'fact2',description ='fact2 about the topic'),
    ResponseSchema(name = 'fact3',description ='fact3 about the topic')
]

parsers = Structuredoutputparsers.from_response_schemas(schema)

template = PromptTemplate(
    template = 'give me 5 facts about {topic} \n {structured_istructions}',
    input_variables = [topic],
    partial_variables = {structured_istructions : parsers.get_format_instructions()}
)

prompt = template.invoke({'topic' : 'black hole'})

result = model.invoke(prompt)

final_result = parsers.parse(result.content)

print(final_result)



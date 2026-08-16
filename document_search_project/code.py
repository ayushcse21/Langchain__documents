#import libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st



load_dotenv()

st.header('Ask me')

#load the document
loader = PyPDFLoader("E:/new_langchain_models/sample.pdf")

docs = loader.lazy_load()

#print(len(docs))

#split the document

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

print(len(chunks))

embedding_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

#create vector store
vector_store = Chroma.from_documents(
    embedding=embedding_model,
    documents=chunks
)

#create retriever
retriever = vector_store.as_retriever(search_kwargs ={'k' : 2})

#create model

llm = HuggingFaceEndpoint(
     repo_id="deepseek-ai/DeepSeek-R1",
     task='text-generation',
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


)

model = ChatHuggingFace(llm = llm)

#write prompt
prompt = PromptTemplate(
    template="""Answer the given query : {query} \n on the basis of following text \n {result} """,
    input_variables=['query','result']
)

# write query
query = st.text_input('Enter your query')

#merge the chunks
def merge_chunk(result):
    text = "\n\n".join(chunks.page_content for chunks in result)
    return text



if st.button('Summarize'):
    
    #retrive relevant documents
    result = retriever.invoke(query)

    result = merge_chunk(result)

    #create final prompt
    prompt = prompt.invoke({'query':query,'result' : result})

    result = model.invoke(prompt)

    #print(query)

    st.write(result.content)

    


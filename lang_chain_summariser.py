from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = OpenAI()

loader = TextLoader("long_text.txt")
documents = loader.load()
long_text = " ".join([document.page_content for document in documents])

prompt_instruction = 'You are a helpful assistant. Please summarize the following text'

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_instruction),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# Invoke the chain and assign the output to a variable
answer = chain.invoke({"input": long_text})

print(answer)

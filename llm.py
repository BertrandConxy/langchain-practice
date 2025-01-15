from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Load .env file
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.stream("write a poem about AI")

for chunk in response:
    print(chunk.content, end="", flush=True)

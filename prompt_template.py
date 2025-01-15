from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load .env file
load_dotenv()

# instantiate the model
llm = ChatOpenAI(model="gpt-4o-mini")

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You make a list of countries that starts with" ),
    ("human", "{input}")
])


# create LLM chain
llm_chain = prompt | llm

response = llm_chain.invoke({"input": "G"})
print(response)